"""Plan Tournament Agent: Generates K plans and runs single-elimination tournament to select the best."""

import copy
import re
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from hashlib import md5
from typing import Any, override

from jinja2 import Template
from loguru import logger

from matharena.api_client import APIClient
from matharena.solvers import BaseAgent, SolverResponse


class PlanTournamentAgent(BaseAgent):
    """
    Agent 1 of the Plan Tournament architecture.

    Generates K plans (textual latents) for a math problem, then runs a
    single-elimination tournament using pairwise comparison to select the best plan.

    The winning plan is stored in history and can be used by ConditionedSolverAgent.
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
        self.RUN_ID = f"plan_tournament_{self.model_config['model'].replace('/', '--')}_{self.problem_idx}_{parameter_hash}"

        # Configuration from scaffold (can be overridden in model config)
        self.n_plans = solver_config.get("n_plans", self.scaffold_config.get("n_plans", 8))
        self.plan_pool_size = self.scaffold_config.get("plan_pool_size", 4)
        self.match_pool_size = self.scaffold_config.get("match_pool_size", 4)

        # Load prompts from scaffold config
        prompts = self.scaffold_config.get("prompts", {})

        planner_prompt = prompts.get("planner", None)
        if planner_prompt is None:
            raise ValueError("Planner prompt not found in scaffold config under 'prompts.planner'")
        self.planner_template = Template(planner_prompt)

        plan_critic_prompt = prompts.get("plan_critic", None)
        if plan_critic_prompt is None:
            raise ValueError("Plan critic prompt not found in scaffold config under 'prompts.plan_critic'")
        self.plan_critic_template = Template(plan_critic_prompt)

        # Create API client
        simple_client_args = copy.deepcopy(default_api_client_args)
        for key in ["human_readable_id", "date", "other_params"]:
            simple_client_args.pop(key, None)
        self.client = APIClient(**simple_client_args)

    def _query_with_cost(self, client: APIClient, query: list[dict[str, Any]]) -> tuple[list[dict], dict]:
        """Returns (conversation, cost_dict) directly from API response."""
        start_time = time.time()
        ret = list(
            client.run_queries(
                [query], no_tqdm=True, custom_indices=[self.batch_idx], ignore_tool_calls=False
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

        return conversation, call_cost

    def _generate_one_plan(self) -> tuple[str, list[dict], dict]:
        """Generate a single plan. Returns (plan, conversation, cost_dict)."""
        prompt = self.planner_template.render(problem=self.stmt)
        convo = [{"role": "user", "content": prompt}]
        convo, call_cost = self._query_with_cost(self.client, convo)
        plan = convo[-1]["content"]
        return plan, convo, call_cost

    def _generate_plans(self) -> tuple[list[str], list[list[dict]], list[dict]]:
        """Generate K plans in parallel. Returns (plans, convos, costs)."""
        plans = []
        convos = []
        costs = []

        with ThreadPoolExecutor(max_workers=self.plan_pool_size) as executor:
            futures = [executor.submit(self._generate_one_plan) for _ in range(self.n_plans)]
            for future in as_completed(futures):
                plan, convo, call_cost = future.result()
                plans.append(plan)
                convos.append(convo)
                costs.append(call_cost)

        return plans, convos, costs

    def _parse_winner(self, response: str) -> str:
        """
        Parse the winner from the critic response.
        Expected format: "Winner: Plan A" or "Winner: Plan B"
        Returns: "A" or "B", defaults to "A" on parse failure.
        """
        # Look for "Winner: Plan A" or "Winner: Plan B"
        match = re.search(r'Winner:\s*Plan\s*([AB])', response, re.IGNORECASE)
        if match:
            return match.group(1).upper()

        # Fallback: look for just "A" or "B" after "Winner:"
        match = re.search(r'Winner:\s*([AB])', response, re.IGNORECASE)
        if match:
            return match.group(1).upper()

        logger.warning(f"[{self.bi}] Failed to parse winner from: {response[-200:]}")
        return "A"  # Default fallback

    def _compare_plans(self, plan_a: str, plan_b: str) -> tuple[str, str, list[dict], dict]:
        """
        Compare two plans using the critic.
        Returns (winner "A" or "B", reasoning, conversation, cost_dict).
        """
        prompt = self.plan_critic_template.render(
            problem=self.stmt,
            plan_a=plan_a,
            plan_b=plan_b
        )
        convo = [{"role": "user", "content": prompt}]
        convo, call_cost = self._query_with_cost(self.client, convo)

        response = convo[-1]["content"]
        winner = self._parse_winner(response)

        # Extract reasoning (everything before "Winner:")
        reasoning_match = re.search(r'^(.*?)Winner:', response, re.DOTALL | re.IGNORECASE)
        reasoning = reasoning_match.group(1).strip() if reasoning_match else response

        return winner, reasoning, convo, call_cost

    def _run_one_match(self, plans: list[str], idx_a: int, idx_b: int, round_num: int, match_num: int) -> dict:
        """Run a single match and return match result dict."""
        winner, reasoning, convo, call_cost = self._compare_plans(plans[idx_a], plans[idx_b])
        winner_idx = idx_a if winner == "A" else idx_b

        match_result = {
            "a": idx_a,
            "b": idx_b,
            "winner": winner,
            "winner_idx": winner_idx,
            "reasoning": reasoning,
            "cost": call_cost,
        }

        # Record in history
        self._add_history(
            step=f"round_{round_num}_match_{match_num}",
            timestep=2,
            conversation=convo,
            plan_a_idx=idx_a,
            plan_b_idx=idx_b,
            winner=winner,
            winner_idx=winner_idx,
            reasoning=reasoning,
            cost=call_cost,
        )

        return match_result

    def _run_tournament(self, plans: list[str]) -> tuple[int, str, dict]:
        """
        Run single-elimination tournament.
        Returns (winner_idx, winning_plan, bracket).
        """
        current_round = list(range(len(plans)))
        bracket = {"rounds": []}
        round_num = 0

        while len(current_round) > 1:
            next_round = []
            round_matches = []
            match_futures = []

            # Run matches in parallel within this round
            with ThreadPoolExecutor(max_workers=self.match_pool_size) as executor:
                match_num = 0
                for i in range(0, len(current_round) - 1, 2):
                    idx_a, idx_b = current_round[i], current_round[i + 1]
                    future = executor.submit(
                        self._run_one_match, plans, idx_a, idx_b, round_num, match_num
                    )
                    match_futures.append((future, match_num))
                    match_num += 1

                # Collect results
                for future, _ in match_futures:
                    match_result = future.result()
                    next_round.append(match_result["winner_idx"])
                    round_matches.append({
                        "a": match_result["a"],
                        "b": match_result["b"],
                        "winner_idx": match_result["winner_idx"],
                    })

            # Handle bye (odd number of contestants)
            if len(current_round) % 2 == 1:
                bye_idx = current_round[-1]
                next_round.append(bye_idx)
                round_matches.append({"bye": bye_idx})
                logger.debug(f"[{self.bi}] Round {round_num}: Plan {bye_idx} gets a bye.")

            bracket["rounds"].append(round_matches)
            logger.debug(f"[{self.bi}] Round {round_num} complete: {len(current_round)} -> {len(next_round)} plans remaining.")
            current_round = next_round
            round_num += 1

        winner_idx = current_round[0]
        return winner_idx, plans[winner_idx], bracket

    @override
    def solve(self, stmt: str) -> SolverResponse:
        """
        Main solve method:
        1. Generate K plans in parallel
        2. Run single-elimination tournament
        3. Return winning plan as the final response
        """
        self._start_run(stmt)
        self._load_checkpoint_if_exists()

        # Phase 1: Generate plans
        if self._history_has_step("plan_generation_summary"):
            logger.debug(f"[{self.bi}] Loading plans from checkpoint.")
            gen_summary = self.get_history_step("plan_generation_summary")
            plans = gen_summary["plans"]
        else:
            logger.debug(f"[{self.bi}] Generating {self.n_plans} plans.")
            plans, gen_convos, gen_costs = self._generate_plans()

            # Record each plan generation with its cost
            for i, (convo, cost) in enumerate(zip(gen_convos, gen_costs)):
                self._add_history(
                    step=f"plan_{i}",
                    timestep=1,
                    conversation=convo,
                    plan_index=i,
                    plan=plans[i],
                    cost=cost,
                )

            # Calculate total generation cost
            total_gen_cost = {
                "cost": sum(c["cost"] for c in gen_costs),
                "input_tokens": sum(c["input_tokens"] for c in gen_costs),
                "output_tokens": sum(c["output_tokens"] for c in gen_costs),
                "time": sum(c["time"] for c in gen_costs),
            }

            # Summary step
            self._add_history(
                step="plan_generation_summary",
                timestep=1,
                conversation=[],
                k_plans=len(plans),
                plans=plans,
                total_cost=total_gen_cost,
            )
            self._save_checkpoint()
            logger.debug(f"[{self.bi}] Generated {len(plans)} plans. Cost: ${total_gen_cost['cost']:.4f}")

        # Phase 2: Tournament
        if self._history_has_step("tournament_summary"):
            logger.debug(f"[{self.bi}] Loading tournament results from checkpoint.")
            tournament_summary = self.get_history_step("tournament_summary")
            winner_idx = tournament_summary["winning_plan_idx"]
            winning_plan = tournament_summary["winning_plan"]
        else:
            logger.debug(f"[{self.bi}] Running plan tournament.")
            winner_idx, winning_plan, bracket = self._run_tournament(plans)

            # Calculate total tournament cost from match history entries
            tournament_cost = {"cost": 0, "input_tokens": 0, "output_tokens": 0, "time": 0}
            for entry in self.history:
                if entry["step"].startswith("round_"):
                    cost = entry.get("cost", {})
                    tournament_cost["cost"] += cost.get("cost", 0)
                    tournament_cost["input_tokens"] += cost.get("input_tokens", 0)
                    tournament_cost["output_tokens"] += cost.get("output_tokens", 0)
                    tournament_cost["time"] += cost.get("time", 0)

            # Summary step
            self._add_history(
                step="tournament_summary",
                timestep=2,
                conversation=[],
                bracket=bracket,
                winning_plan_idx=winner_idx,
                winning_plan=winning_plan,
                total_cost=tournament_cost,
            )
            self._save_checkpoint()
            logger.info(f"[{self.bi}] Tournament complete. Winner: Plan {winner_idx}. Cost: ${tournament_cost['cost']:.4f}")

        # Build final conversation (problem + winning plan)
        final_convo = [
            {"role": "user", "content": self.default_prompt_template.format(problem=stmt)},
            {"role": "assistant", "content": winning_plan}
        ]

        return self._end_run(final_convo)
