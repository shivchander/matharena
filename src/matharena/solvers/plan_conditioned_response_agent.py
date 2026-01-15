"""Plan Conditioned Response Agent: Generates a single solution conditioned on a scored plan."""

import copy
import json
import os
import time
from hashlib import md5
from typing import Any, override

from jinja2 import Template
from loguru import logger

from matharena.api_client import APIClient
from matharena.solvers import BaseAgent, SolverResponse


class PlanConditionedResponseAgent(BaseAgent):
    """
    Agent that reads the best plan from an upstream PlanScoringAgent,
    and generates ONE solution conditioned on it.

    Use --n M at the runner level to generate M solutions (M separate runs).
    Each run produces one solution stored in the output JSON.

    Downstream agents (ResponseScoringAgent) read all M solutions from messages[0..M-1].
    """

    def __init__(self, batch_idx, problem_idx, run_idx, solver_config,
                 default_prompt_template, default_api_client_args):
        super().__init__(batch_idx, problem_idx, run_idx, solver_config,
                         default_prompt_template, default_api_client_args)

        self.model_config = solver_config["model_config"]
        self.scaffold_config = solver_config["scaffold_config"]

        # Create a unique run ID for checkpointing
        stringify_params = str(self.model_config) + str(self.scaffold_config)
        parameter_hash = md5(stringify_params.encode('utf-8')).hexdigest()[:8]
        self.RUN_ID = f"plan_cond_resp_{self.model_config['model'].replace('/', '--')}_{self.problem_idx}_{parameter_hash}"

        # Temperature support
        self.temperature = solver_config.get(
            "temperature",
            self.scaffold_config.get("temperature", None)
        )

        # Plan score source model (where to read scored plans from)
        self.plan_score_source_model = solver_config.get(
            "plan_score_source_model",
            self.scaffold_config.get("plan_score_source_model", None)
        )
        if self.plan_score_source_model is None:
            raise ValueError("plan_score_source_model must be specified in solver config or scaffold config")

        # Competition name (for building output path)
        self.competition = solver_config.get("competition", None)
        if self.competition is None:
            raise ValueError("competition must be specified in solver config")

        # Load conditioned solver prompt from scaffold config
        conditioned_solver_prompt = self.scaffold_config.get("prompts", {}).get("conditioned_solver", None)
        if conditioned_solver_prompt is None:
            raise ValueError("Conditioned solver prompt not found in scaffold config under 'prompts.conditioned_solver'")
        self.conditioned_solver_template = Template(conditioned_solver_prompt)

        # Create API client with temperature if specified
        simple_client_args = copy.deepcopy(default_api_client_args)
        for key in ["human_readable_id", "date", "other_params"]:
            simple_client_args.pop(key, None)

        if self.temperature is not None:
            simple_client_args["temperature"] = self.temperature

        self.client = APIClient(**simple_client_args)

    def _load_best_plan(self) -> tuple[str, int, int]:
        """
        Load the best plan from upstream PlanScoringAgent output.
        Returns (best_plan, best_plan_idx, best_score)
        """
        score_output_path = os.path.join(
            "outputs",
            self.competition,
            self.plan_score_source_model,
            f"{self.problem_idx}.json"
        )

        if not os.path.exists(score_output_path):
            raise FileNotFoundError(
                f"Plan score source not found at {score_output_path}. "
                f"Run {self.plan_score_source_model} first for problem {self.problem_idx}"
            )

        with open(score_output_path, "r") as f:
            data = json.load(f)

        # Find scoring_summary in history (first run)
        history = data.get("history", [[]])
        if not history or not history[0]:
            raise ValueError(f"No history found in {score_output_path}")

        for step in history[0]:
            if step.get("step") == "scoring_summary":
                best_plan = step.get("best_plan")
                best_plan_idx = step.get("best_plan_idx")
                best_score = step.get("best_score")
                if best_plan is not None:
                    return best_plan, best_plan_idx, best_score

        raise ValueError(f"No scoring_summary found in {score_output_path}")

    def _generate_solution(self, plan: str) -> tuple[str, list[dict], dict]:
        """Generate a single solution conditioned on the plan. Returns (solution, conversation, cost_dict)."""
        prompt = self.conditioned_solver_template.render(
            problem=self.stmt,
            plan=plan
        )
        convo = [{"role": "user", "content": prompt}]

        start_time = time.time()
        ret = list(
            self.client.run_queries(
                [convo], no_tqdm=True, custom_indices=[self.batch_idx], ignore_tool_calls=False
            )
        )
        _, conversation, detailed_cost = ret[0]
        elapsed_time = time.time() - start_time

        with self._lock:
            self.detailed_cost["cost"] += detailed_cost["cost"]
            self.detailed_cost["input_tokens"] += detailed_cost["input_tokens"]
            self.detailed_cost["output_tokens"] += detailed_cost["output_tokens"]
            self.detailed_cost["time"] += elapsed_time

        call_cost = {
            "cost": detailed_cost["cost"],
            "input_tokens": detailed_cost["input_tokens"],
            "output_tokens": detailed_cost["output_tokens"],
            "time": elapsed_time,
        }

        solution = conversation[-1]["content"]
        return solution, conversation, call_cost

    @override
    def solve(self, stmt: str) -> SolverResponse:
        """
        Generate ONE solution conditioned on the best plan.
        Use --n M to generate M solutions (M separate runs).
        """
        self._start_run(stmt)
        self._load_checkpoint_if_exists()

        # Step 1: Load best plan from upstream
        if self._history_has_step("plan_loaded"):
            logger.debug(f"[{self.bi}] Loading plan from checkpoint.")
            plan_loaded = self.get_history_step("plan_loaded")
            best_plan = plan_loaded["best_plan"]
            best_plan_idx = plan_loaded["best_plan_idx"]
        else:
            logger.debug(f"[{self.bi}] Loading best plan from {self.plan_score_source_model}.")
            best_plan, best_plan_idx, best_score = self._load_best_plan()

            self._add_history(
                step="plan_loaded",
                timestep=1,
                conversation=[],
                plan_score_source_model=self.plan_score_source_model,
                best_plan=best_plan,
                best_plan_idx=best_plan_idx,
                best_score=best_score,
            )
            self._save_checkpoint()
            logger.debug(f"[{self.bi}] Loaded best plan #{best_plan_idx} with score {best_score}/10.")

        # Step 2: Generate solution conditioned on best plan
        if self._history_has_step("solution_generated"):
            logger.debug(f"[{self.bi}] Loading solution from checkpoint.")
            solution_step = self.get_history_step("solution_generated")
            solution = solution_step["solution"]
            convo = solution_step["messages"]
        else:
            logger.debug(f"[{self.bi}] Generating solution (run {self.run_idx}) conditioned on plan #{best_plan_idx}.")
            solution, convo, call_cost = self._generate_solution(best_plan)

            self._add_history(
                step="solution_generated",
                timestep=2,
                conversation=convo,
                solution=solution,
                plan_used=best_plan,
                plan_idx=best_plan_idx,
                run_idx=self.run_idx,
                cost=call_cost,
            )
            self._save_checkpoint()
            logger.debug(f"[{self.bi}] Generated solution. Cost: ${call_cost['cost']:.4f}")

        logger.info(f"[{self.bi}] Solution generation complete (run {self.run_idx}).")

        # Return the solution as the final response
        final_convo = [
            {"role": "user", "content": self.conditioned_solver_template.render(problem=stmt, plan=best_plan)},
            {"role": "assistant", "content": solution}
        ]

        return self._end_run(final_convo)
