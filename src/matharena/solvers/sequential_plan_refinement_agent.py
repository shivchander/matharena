"""Sequential Plan Refinement Agent: Iteratively critiques and refines plans with early stopping."""

import copy
import re
import time
from hashlib import md5
from typing import Any, override

from jinja2 import Template
from loguru import logger

from matharena.api_client import APIClient
from matharena.solvers import BaseAgent, SolverResponse


class SequentialPlanRefinementAgent(BaseAgent):
    """
    An agent that generates an initial plan and then iteratively critiques and refines it
    through up to max_refinements rounds, with early stopping when the plan is sufficient.

    Workflow:
    1. Generate initial plan
    2. For each refinement iteration (up to max_refinements):
       - Critique the current plan
       - Decide: STOP (plan is sufficient) or CONTINUE (needs refinement)
       - If CONTINUE, generate improved plan
    3. Return final refined plan
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
        self.RUN_ID = f"seq_plan_refine_{self.model_config['model'].replace('/', '--')}_{self.problem_idx}_{parameter_hash}"

        # Configuration from scaffold
        self.max_refinements = self.scaffold_config.get("max_refinements", 8)

        # Load prompts from scaffold config
        prompts = self.scaffold_config.get("prompts", {})

        initial_planner_prompt = prompts.get("initial_planner", None)
        if initial_planner_prompt is None:
            raise ValueError("Initial planner prompt not found in scaffold config under 'prompts.initial_planner'")
        self.initial_planner_template = Template(initial_planner_prompt)

        critic_and_refine_prompt = prompts.get("critic_and_refine", None)
        if critic_and_refine_prompt is None:
            raise ValueError("Critic and refine prompt not found in scaffold config under 'prompts.critic_and_refine'")
        self.critic_and_refine_template = Template(critic_and_refine_prompt)

        # Create API client
        simple_client_args = copy.deepcopy(default_api_client_args)
        # Remove non-API client args
        for key in ["human_readable_id", "date", "other_params"]:
            simple_client_args.pop(key, None)
        self.client = APIClient(**simple_client_args)

    def _query_with_cost(self, client: APIClient, query: list[dict[str, Any]]) -> tuple[list[dict], dict]:
        """
        A wrapper that runs a query and returns both the conversation and the per-call cost.
        Returns: (conversation, cost_dict) where cost_dict has cost, input_tokens, output_tokens, time.
        """
        start_time = time.time()
        ret = list(
            client.run_queries(
                [query], no_tqdm=True, custom_indices=[self.batch_idx], ignore_tool_calls=False
            )
        )
        _, conversation, detailed_cost = ret[0]
        elapsed_time = time.time() - start_time

        # Update the cumulative cost (thread-safe)
        with self._lock:
            self.detailed_cost["cost"] += detailed_cost["cost"]
            self.detailed_cost["input_tokens"] += detailed_cost["input_tokens"]
            self.detailed_cost["output_tokens"] += detailed_cost["output_tokens"]
            self.detailed_cost["time"] += elapsed_time

        # Return the per-call cost
        call_cost = {
            "cost": detailed_cost["cost"],
            "input_tokens": detailed_cost["input_tokens"],
            "output_tokens": detailed_cost["output_tokens"],
            "time": elapsed_time,
        }

        return conversation, call_cost

    def _generate_initial_plan(self) -> tuple[str, list[dict], dict]:
        """Generate the initial plan. Returns (plan, conversation, cost_dict)."""
        prompt = self.initial_planner_template.render(problem=self.stmt)
        convo = [{"role": "user", "content": prompt}]
        convo, call_cost = self._query_with_cost(self.client, convo)
        plan = convo[-1]["content"]
        return plan, convo, call_cost

    def _parse_critique_response(self, response: str) -> tuple[str, str, str]:
        """
        Parse the critique response into (critique, decision, refinement).

        Expected format:
        CRITIQUE: [analysis]
        DECISION: [STOP/CONTINUE]
        REFINEMENT: [improved plan if CONTINUE]

        Returns:
            critique: The critique text
            decision: "STOP" or "CONTINUE"
            refinement: The refined plan (empty string if STOP)
        """
        # Extract CRITIQUE
        critique_match = re.search(r'CRITIQUE:\s*(.*?)(?=DECISION:|$)', response, re.DOTALL | re.IGNORECASE)
        critique = critique_match.group(1).strip() if critique_match else ""

        # Extract DECISION
        decision_match = re.search(r'DECISION:\s*(STOP|CONTINUE)', response, re.IGNORECASE)
        decision = decision_match.group(1).upper() if decision_match else "STOP"

        # Extract REFINEMENT (only if CONTINUE)
        refinement = ""
        if decision == "CONTINUE":
            refinement_match = re.search(r'REFINEMENT:\s*(.*?)$', response, re.DOTALL | re.IGNORECASE)
            refinement = refinement_match.group(1).strip() if refinement_match else ""

            # If no refinement found but decision is CONTINUE, log warning and treat as STOP
            if not refinement:
                logger.warning(f"[{self.bi}] DECISION=CONTINUE but no REFINEMENT found. Treating as STOP.")
                decision = "STOP"

        return critique, decision, refinement

    def _critique_and_refine(self, current_plan: str) -> tuple[str, str, str, list[dict], dict]:
        """
        Critique the current plan and potentially refine it.

        Returns:
            critique: The critique analysis
            decision: "STOP" or "CONTINUE"
            refined_plan: The improved plan (empty if STOP)
            conversation: The full conversation
            cost_dict: The cost of this iteration
        """
        prompt = self.critic_and_refine_template.render(
            problem=self.stmt,
            current_plan=current_plan
        )
        convo = [{"role": "user", "content": prompt}]
        convo, call_cost = self._query_with_cost(self.client, convo)

        response = convo[-1]["content"]
        critique, decision, refinement = self._parse_critique_response(response)

        return critique, decision, refinement, convo, call_cost

    @override
    def solve(self, stmt: str) -> SolverResponse:
        """
        Main solve method:
        1. Generate initial plan
        2. Iteratively critique and refine up to max_refinements times
        3. Early stop when model decides plan is sufficient
        4. Return final refined plan
        """
        self._start_run(stmt)
        self._load_checkpoint_if_exists()

        # Step 1: Generate initial plan
        if self._history_has_step("initial_plan"):
            logger.debug(f"[{self.bi}] Loading initial plan from checkpoint.")
            initial_step = self.get_history_step("initial_plan")
            current_plan = initial_step["plan"]
        else:
            logger.debug(f"[{self.bi}] Generating initial plan.")
            current_plan, init_convo, init_cost = self._generate_initial_plan()

            self._add_history(
                step="initial_plan",
                timestep=1,
                conversation=init_convo,
                plan=current_plan,
                cost=init_cost,
            )
            self._save_checkpoint()
            logger.debug(f"[{self.bi}] Initial plan generated. Cost: ${init_cost['cost']:.4f}")

        # Step 2: Iterative refinement
        refinement_iteration = 0
        stopped_early = False

        for i in range(self.max_refinements):
            refinement_step = f"refinement_{i}"

            # Skip if already done
            if self._history_has_step(refinement_step):
                logger.debug(f"[{self.bi}] Refinement {i} already completed, loading from checkpoint.")
                ref_step = self.get_history_step(refinement_step)
                decision = ref_step["decision"]
                if decision == "STOP":
                    stopped_early = True
                    refinement_iteration = i
                    break
                current_plan = ref_step.get("refined_plan", current_plan)
                refinement_iteration = i + 1
                continue

            # Run critique and refinement
            logger.debug(f"[{self.bi}] Running refinement iteration {i}.")
            critique, decision, refined_plan, ref_convo, ref_cost = self._critique_and_refine(current_plan)

            # Record in history
            self._add_history(
                step=refinement_step,
                timestep=2 + i,
                conversation=ref_convo,
                current_plan=current_plan,
                critique=critique,
                decision=decision,
                refined_plan=refined_plan if decision == "CONTINUE" else "",
                cost=ref_cost,
            )
            self._save_checkpoint()

            logger.debug(f"[{self.bi}] Refinement {i} complete. Decision: {decision}. Cost: ${ref_cost['cost']:.4f}")

            # Check for early stopping
            if decision == "STOP":
                stopped_early = True
                refinement_iteration = i
                logger.info(f"[{self.bi}] Early stopping at refinement {i}. Plan deemed sufficient.")
                break

            # Update current plan for next iteration
            current_plan = refined_plan
            refinement_iteration = i + 1

        # Step 3: Add summary
        if not self._history_has_step("refinement_summary"):
            # Calculate total refinement cost
            total_refinement_cost = {"cost": 0, "input_tokens": 0, "output_tokens": 0, "time": 0}
            for entry in self.history:
                if entry["step"].startswith("refinement_"):
                    cost = entry.get("cost", {})
                    total_refinement_cost["cost"] += cost.get("cost", 0)
                    total_refinement_cost["input_tokens"] += cost.get("input_tokens", 0)
                    total_refinement_cost["output_tokens"] += cost.get("output_tokens", 0)
                    total_refinement_cost["time"] += cost.get("time", 0)

            self._add_history(
                step="refinement_summary",
                timestep=2 + self.max_refinements,
                conversation=[],
                total_iterations=refinement_iteration,
                stopped_early=stopped_early,
                final_plan=current_plan,
                total_cost=total_refinement_cost,
            )
            self._save_checkpoint()
            logger.info(f"[{self.bi}] Refinement complete. Total iterations: {refinement_iteration}. "
                       f"Total cost: ${total_refinement_cost['cost']:.4f}")

        # Build final conversation (user question + final plan)
        final_convo = [
            {"role": "user", "content": self.stmt},
            {"role": "assistant", "content": current_plan}
        ]

        return self._end_run(final_convo)
