"""Plan Generation Agent: Generates a single plan (textual latent) for a math problem."""

import copy
import time
from hashlib import md5
from typing import Any, override

from jinja2 import Template
from loguru import logger

from matharena.api_client import APIClient
from matharena.solvers import BaseAgent, SolverResponse


class PlanGenerationAgent(BaseAgent):
    """
    Agent that generates ONE plan (textual latent) for a math problem.

    Use --n K at the runner level to generate K plans (K separate runs).
    Each run produces one plan stored in the output JSON.

    Downstream agents (PlanScoringAgent) read all K plans from messages[0..K-1].
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
        self.RUN_ID = f"plan_gen_{self.model_config['model'].replace('/', '--')}_{self.problem_idx}_{parameter_hash}"

        # Temperature support (optional, from model config or scaffold config)
        self.temperature = solver_config.get(
            "temperature",
            self.scaffold_config.get("temperature", None)
        )

        # Load planner prompt from scaffold config
        planner_prompt = self.scaffold_config.get("prompts", {}).get("planner", None)
        if planner_prompt is None:
            raise ValueError("Planner prompt not found in scaffold config under 'prompts.planner'")
        self.planner_template = Template(planner_prompt)

        # Create API client with temperature if specified
        simple_client_args = copy.deepcopy(default_api_client_args)
        for key in ["human_readable_id", "date", "other_params"]:
            simple_client_args.pop(key, None)

        # Override temperature if specified in agent config
        if self.temperature is not None:
            simple_client_args["temperature"] = self.temperature

        self.client = APIClient(**simple_client_args)

    def _generate_plan(self) -> tuple[str, list[dict], dict]:
        """Generate a single plan. Returns (plan, conversation, cost_dict)."""
        prompt = self.planner_template.render(problem=self.stmt)
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

        plan = conversation[-1]["content"]
        return plan, conversation, call_cost

    @override
    def solve(self, stmt: str) -> SolverResponse:
        """
        Generate ONE plan for the problem.
        Use --n K to generate K plans (K separate runs).
        """
        self._start_run(stmt)
        self._load_checkpoint_if_exists()

        # Generate plan
        if self._history_has_step("plan_generated"):
            logger.debug(f"[{self.bi}] Loading plan from checkpoint.")
            plan_step = self.get_history_step("plan_generated")
            plan = plan_step["plan"]
            convo = plan_step["messages"]
        else:
            logger.debug(f"[{self.bi}] Generating plan (run {self.run_idx}).")
            plan, convo, call_cost = self._generate_plan()

            self._add_history(
                step="plan_generated",
                timestep=1,
                conversation=convo,
                plan=plan,
                run_idx=self.run_idx,
                cost=call_cost,
            )
            self._save_checkpoint()
            logger.debug(f"[{self.bi}] Generated plan. Cost: ${call_cost['cost']:.4f}")

        logger.info(f"[{self.bi}] Plan generation complete (run {self.run_idx}).")

        # Return the plan as the final response
        final_convo = [
            {"role": "user", "content": self.planner_template.render(problem=stmt)},
            {"role": "assistant", "content": plan}
        ]

        return self._end_run(final_convo)
