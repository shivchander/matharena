"""Batch Plan Generation Agent: Generates N diverse plans in a single API call."""

import copy
import re
import time
from hashlib import md5
from typing import Any, override

from jinja2 import Template
from loguru import logger

from matharena.api_client import APIClient
from matharena.solvers import BaseAgent, SolverResponse


class BatchPlanGenerationAgent(BaseAgent):
    """
    Agent that generates N diverse plans in a SINGLE API call.

    Unlike PlanGenerationAgent which makes N independent calls (via --n N),
    this agent asks the model to produce N explicitly different plans in one request.
    This encourages diversity since the model can contrast strategies.

    Use --n K to generate K batches (each batch produces n_plans plans).
    """

    def __init__(self, batch_idx, problem_idx, run_idx, solver_config,
                 default_prompt_template, default_api_client_args):
        super().__init__(batch_idx, problem_idx, run_idx, solver_config,
                         default_prompt_template, default_api_client_args)

        self.model_config = solver_config["model_config"]
        self.scaffold_config = solver_config["scaffold_config"]

        # Number of plans to generate in a single call
        # Model config overrides scaffold config
        self.n_plans = solver_config.get(
            "n_plans",
            self.scaffold_config.get("n_plans", 16)
        )

        # Create a unique run ID for checkpointing
        stringify_params = str(self.model_config) + str(self.scaffold_config) + str(self.n_plans)
        parameter_hash = md5(stringify_params.encode('utf-8')).hexdigest()[:8]
        self.RUN_ID = f"batch_plan_gen_{self.model_config['model'].replace('/', '--')}_{self.problem_idx}_{parameter_hash}"

        # Temperature support (optional, from model config or scaffold config)
        self.temperature = solver_config.get(
            "temperature",
            self.scaffold_config.get("temperature", None)
        )

        # Load batch planner prompt from scaffold config
        batch_planner_prompt = self.scaffold_config.get("prompts", {}).get("batch_planner", None)
        if batch_planner_prompt is None:
            raise ValueError("Batch planner prompt not found in scaffold config under 'prompts.batch_planner'")
        self.batch_planner_template = Template(batch_planner_prompt)

        # Create API client with temperature if specified
        simple_client_args = copy.deepcopy(default_api_client_args)
        for key in ["human_readable_id", "date", "other_params"]:
            simple_client_args.pop(key, None)

        # Override temperature if specified in agent config
        if self.temperature is not None:
            simple_client_args["temperature"] = self.temperature

        self.client = APIClient(**simple_client_args)

    def _parse_plans(self, response: str) -> list[str]:
        """
        Parse N plans from a single response using 'Plan:' delimiter.

        Returns a list of plan strings, each starting with "Plan:" and containing
        the numbered steps.
        """
        # Split by "Plan:" but keep the delimiter for each plan
        # Pattern: newline (optional whitespace) Plan: (optional whitespace)
        parts = re.split(r'\n\s*Plan:\s*', response)

        plans = []
        for i, part in enumerate(parts):
            part = part.strip()
            if not part:
                continue

            # Reconstruct the plan with "Plan:" prefix
            if i == 0 and response.strip().startswith("Plan:"):
                # First part already has Plan: at the start of response
                plan = "Plan:\n" + part.split('\n', 1)[1] if '\n' in part else "Plan:\n" + part
            elif i > 0:
                # Subsequent parts need Plan: prefix added back
                plan = "Plan:\n" + part
            else:
                # First part doesn't start with Plan:, skip any preamble
                continue

            plans.append(plan)

        # Fallback: if parsing failed, try a simpler approach
        if len(plans) == 0:
            # Try splitting by numbered plan headers like "Plan 1:", "Plan 2:", etc.
            parts = re.split(r'\n\s*Plan\s*\d+:\s*', response)
            for i, part in enumerate(parts):
                part = part.strip()
                if part:
                    plans.append(f"Plan:\n{part}")

        return plans

    def _generate_batch_plans(self) -> tuple[list[str], list[dict], dict]:
        """
        Generate N plans in a single API call.

        Returns:
            plans: List of plan strings
            conversation: The full conversation with the model
            cost_dict: Cost information for this call
        """
        prompt = self.batch_planner_template.render(
            problem=self.stmt,
            n_plans=self.n_plans
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

        # Parse multiple plans from the response
        response_text = conversation[-1]["content"]
        plans = self._parse_plans(response_text)

        if len(plans) < self.n_plans:
            logger.warning(
                f"[{self.bi}] Expected {self.n_plans} plans but parsed {len(plans)}. "
                f"Response may not have followed the format correctly."
            )
        elif len(plans) > self.n_plans:
            logger.warning(
                f"[{self.bi}] Expected {self.n_plans} plans but parsed {len(plans)}. "
                f"Truncating to first {self.n_plans}."
            )
            plans = plans[:self.n_plans]

        return plans, conversation, call_cost

    @override
    def solve(self, stmt: str) -> SolverResponse:
        """
        Generate N diverse plans for the problem in a single API call.

        The plans are stored in the history as separate entries, compatible
        with downstream PlanScoringAgent.
        """
        self._start_run(stmt)
        self._load_checkpoint_if_exists()

        # Generate batch of plans
        if self._history_has_step("batch_plans_generated"):
            logger.debug(f"[{self.bi}] Loading plans from checkpoint.")
            plan_step = self.get_history_step("batch_plans_generated")
            plans = plan_step["plans"]
            convo = plan_step["messages"]
        else:
            logger.debug(f"[{self.bi}] Generating {self.n_plans} diverse plans (batch run {self.run_idx}).")
            plans, convo, call_cost = self._generate_batch_plans()

            self._add_history(
                step="batch_plans_generated",
                timestep=1,
                conversation=convo,
                plans=plans,
                n_plans_requested=self.n_plans,
                n_plans_parsed=len(plans),
                run_idx=self.run_idx,
                cost=call_cost,
            )
            self._save_checkpoint()
            logger.debug(f"[{self.bi}] Generated {len(plans)} plans. Cost: ${call_cost['cost']:.4f}")

        logger.info(f"[{self.bi}] Batch plan generation complete (run {self.run_idx}): {len(plans)} plans.")

        # Create separate history entries for each plan (for downstream compatibility)
        # This matches the format that PlanScoringAgent expects
        for i, plan in enumerate(plans):
            self._add_history(
                step="plan_generated",
                timestep=2 + i,
                conversation=[
                    {"role": "user", "content": self.batch_planner_template.render(problem=stmt, n_plans=self.n_plans)},
                    {"role": "assistant", "content": plan}
                ],
                plan=plan,
                plan_index=i,
                run_idx=self.run_idx,
            )

        # Return the first plan as the primary response
        # (all plans are accessible via history for downstream agents)
        primary_plan = plans[0] if plans else "No plans generated."
        final_convo = [
            {"role": "user", "content": self.batch_planner_template.render(problem=stmt, n_plans=self.n_plans)},
            {"role": "assistant", "content": primary_plan}
        ]

        return self._end_run(final_convo)
