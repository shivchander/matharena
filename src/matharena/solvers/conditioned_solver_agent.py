"""Conditioned Solver Agent: Generates solutions conditioned on a plan and runs tournament to select best."""

import copy
import json
import os
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


class ConditionedSolverAgent(BaseAgent):
    """
    Agent 2 of the Plan Tournament architecture.

    Loads the winning plan from PlanTournamentAgent's output, generates M solutions
    conditioned on that plan, then runs a single-elimination tournament to select the best.
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
        self.RUN_ID = f"conditioned_solver_{self.model_config['model'].replace('/', '--')}_{self.problem_idx}_{parameter_hash}"

        # Configuration from scaffold (can be overridden in model config)
        self.n_solutions = solver_config.get("n_solutions", self.scaffold_config.get("n_solutions", 4))
        self.solution_pool_size = self.scaffold_config.get("solution_pool_size", 4)
        self.match_pool_size = self.scaffold_config.get("match_pool_size", 4)

        # Path to plan source (Agent 1's output)
        self.plan_source_model = solver_config.get(
            "plan_source_model",
            self.scaffold_config.get("plan_source_model")
        )
        if self.plan_source_model is None:
            raise ValueError("plan_source_model not specified in model or scaffold config")

        # Competition info (will be set via solve())
        self.competition = solver_config.get("competition", None)

        # Load prompts from scaffold config
        prompts = self.scaffold_config.get("prompts", {})

        conditioned_solver_prompt = prompts.get("conditioned_solver", None)
        if conditioned_solver_prompt is None:
            raise ValueError("Conditioned solver prompt not found in scaffold config under 'prompts.conditioned_solver'")
        self.conditioned_solver_template = Template(conditioned_solver_prompt)

        solution_critic_prompt = prompts.get("solution_critic", None)
        if solution_critic_prompt is None:
            raise ValueError("Solution critic prompt not found in scaffold config under 'prompts.solution_critic'")
        self.solution_critic_template = Template(solution_critic_prompt)

        # Create API client
        simple_client_args = copy.deepcopy(default_api_client_args)
        for key in ["human_readable_id", "date", "other_params"]:
            simple_client_args.pop(key, None)
        self.client = APIClient(**simple_client_args)

    def _load_winning_plan(self) -> str:
        """
        Load the winning plan from PlanTournamentAgent's output JSON.

        Looks in outputs/{competition}/{plan_source_model}/{problem_idx}.json
        """
        # Build path to plan source output
        # plan_source_model format: "openai/gpt-52--none-plan-tournament"
        plan_output_path = os.path.join(
            "outputs",
            self.competition,
            self.plan_source_model,
            f"{self.problem_idx}.json"
        )

        if not os.path.exists(plan_output_path):
            raise FileNotFoundError(
                f"Plan source not found at {plan_output_path}. "
                f"Run PlanTournamentAgent first with model {self.plan_source_model}"
            )

        with open(plan_output_path, "r") as f:
            plan_output = json.load(f)

        # Find the winning plan in history
        # History is a list of runs, each run is a list of steps
        # We want the first run's tournament_summary
        history = plan_output.get("history", [[]])
        if not history or not history[0]:
            raise ValueError(f"No history found in {plan_output_path}")

        run_history = history[0]  # First run
        for step in run_history:
            if step.get("step") == "tournament_summary":
                winning_plan = step.get("winning_plan")
                if winning_plan:
                    logger.info(f"[{self.bi}] Loaded winning plan from {plan_output_path}")
                    return winning_plan

        raise ValueError(f"No tournament_summary with winning_plan found in {plan_output_path}")

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

    def _generate_one_solution(self, plan: str) -> tuple[str, list[dict], dict]:
        """Generate a single solution conditioned on plan. Returns (solution, conversation, cost_dict)."""
        prompt = self.conditioned_solver_template.render(
            problem=self.stmt,
            plan=plan
        )
        convo = [{"role": "user", "content": prompt}]
        convo, call_cost = self._query_with_cost(self.client, convo)
        solution = convo[-1]["content"]
        return solution, convo, call_cost

    def _generate_solutions(self, plan: str) -> tuple[list[str], list[list[dict]], list[dict]]:
        """Generate M solutions in parallel. Returns (solutions, convos, costs)."""
        solutions = []
        convos = []
        costs = []

        with ThreadPoolExecutor(max_workers=self.solution_pool_size) as executor:
            futures = [executor.submit(self._generate_one_solution, plan) for _ in range(self.n_solutions)]
            for future in as_completed(futures):
                solution, convo, call_cost = future.result()
                solutions.append(solution)
                convos.append(convo)
                costs.append(call_cost)

        return solutions, convos, costs

    def _parse_winner(self, response: str) -> str:
        """
        Parse the winner from the critic response.
        Expected format: "Winner: Solution A" or "Winner: Solution B"
        Returns: "A" or "B", defaults to "A" on parse failure.
        """
        # Look for "Winner: Solution A" or "Winner: Solution B"
        match = re.search(r'Winner:\s*Solution\s*([AB])', response, re.IGNORECASE)
        if match:
            return match.group(1).upper()

        # Fallback: look for just "A" or "B" after "Winner:"
        match = re.search(r'Winner:\s*([AB])', response, re.IGNORECASE)
        if match:
            return match.group(1).upper()

        logger.warning(f"[{self.bi}] Failed to parse solution winner from: {response[-200:]}")
        return "A"  # Default fallback

    def _compare_solutions(self, sol_a: str, sol_b: str) -> tuple[str, str, list[dict], dict]:
        """
        Compare two solutions using the critic.
        Returns (winner "A" or "B", reasoning, conversation, cost_dict).
        """
        prompt = self.solution_critic_template.render(
            problem=self.stmt,
            solution_a=sol_a,
            solution_b=sol_b
        )
        convo = [{"role": "user", "content": prompt}]
        convo, call_cost = self._query_with_cost(self.client, convo)

        response = convo[-1]["content"]
        winner = self._parse_winner(response)

        # Extract reasoning (everything before "Winner:")
        reasoning_match = re.search(r'^(.*?)Winner:', response, re.DOTALL | re.IGNORECASE)
        reasoning = reasoning_match.group(1).strip() if reasoning_match else response

        return winner, reasoning, convo, call_cost

    def _run_one_match(self, solutions: list[str], idx_a: int, idx_b: int, round_num: int, match_num: int) -> dict:
        """Run a single match and return match result dict."""
        winner, reasoning, convo, call_cost = self._compare_solutions(solutions[idx_a], solutions[idx_b])
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
            step=f"sol_round_{round_num}_match_{match_num}",
            timestep=2,
            conversation=convo,
            sol_a_idx=idx_a,
            sol_b_idx=idx_b,
            winner=winner,
            winner_idx=winner_idx,
            reasoning=reasoning,
            cost=call_cost,
        )

        return match_result

    def _run_tournament(self, solutions: list[str]) -> tuple[int, str, dict]:
        """
        Run single-elimination tournament on solutions.
        Returns (winner_idx, winning_solution, bracket).
        """
        current_round = list(range(len(solutions)))
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
                        self._run_one_match, solutions, idx_a, idx_b, round_num, match_num
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
                logger.debug(f"[{self.bi}] Solution round {round_num}: Solution {bye_idx} gets a bye.")

            bracket["rounds"].append(round_matches)
            logger.debug(f"[{self.bi}] Solution round {round_num} complete: {len(current_round)} -> {len(next_round)} solutions remaining.")
            current_round = next_round
            round_num += 1

        winner_idx = current_round[0]
        return winner_idx, solutions[winner_idx], bracket

    @override
    def solve(self, stmt: str) -> SolverResponse:
        """
        Main solve method:
        1. Load winning plan from PlanTournamentAgent output
        2. Generate M solutions conditioned on the plan
        3. Run single-elimination tournament on solutions
        4. Return winning solution as the final response
        """
        self._start_run(stmt)
        self._load_checkpoint_if_exists()

        # Phase 1: Load winning plan
        if self._history_has_step("plan_loaded"):
            logger.debug(f"[{self.bi}] Loading plan from checkpoint.")
            plan_step = self.get_history_step("plan_loaded")
            winning_plan = plan_step["winning_plan"]
        else:
            logger.debug(f"[{self.bi}] Loading winning plan from {self.plan_source_model}.")
            winning_plan = self._load_winning_plan()

            self._add_history(
                step="plan_loaded",
                timestep=1,
                conversation=[],
                plan_source_model=self.plan_source_model,
                winning_plan=winning_plan,
            )
            self._save_checkpoint()
            logger.debug(f"[{self.bi}] Loaded winning plan ({len(winning_plan)} chars).")

        # Phase 2: Generate solutions
        if self._history_has_step("solution_generation_summary"):
            logger.debug(f"[{self.bi}] Loading solutions from checkpoint.")
            gen_summary = self.get_history_step("solution_generation_summary")
            solutions = gen_summary["solutions"]
        else:
            logger.debug(f"[{self.bi}] Generating {self.n_solutions} solutions conditioned on plan.")
            solutions, gen_convos, gen_costs = self._generate_solutions(winning_plan)

            # Record each solution generation with its cost
            for i, (convo, cost) in enumerate(zip(gen_convos, gen_costs)):
                self._add_history(
                    step=f"solution_{i}",
                    timestep=1,
                    conversation=convo,
                    solution_index=i,
                    solution=solutions[i],
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
                step="solution_generation_summary",
                timestep=1,
                conversation=[],
                m_solutions=len(solutions),
                solutions=solutions,
                total_cost=total_gen_cost,
            )
            self._save_checkpoint()
            logger.debug(f"[{self.bi}] Generated {len(solutions)} solutions. Cost: ${total_gen_cost['cost']:.4f}")

        # Phase 3: Solution Tournament
        if self._history_has_step("solution_tournament_summary"):
            logger.debug(f"[{self.bi}] Loading solution tournament results from checkpoint.")
            tournament_summary = self.get_history_step("solution_tournament_summary")
            winner_idx = tournament_summary["winning_solution_idx"]
            winning_solution = tournament_summary["winning_solution"]
        else:
            logger.debug(f"[{self.bi}] Running solution tournament.")
            winner_idx, winning_solution, bracket = self._run_tournament(solutions)

            # Calculate total tournament cost from match history entries
            tournament_cost = {"cost": 0, "input_tokens": 0, "output_tokens": 0, "time": 0}
            for entry in self.history:
                if entry["step"].startswith("sol_round_"):
                    cost = entry.get("cost", {})
                    tournament_cost["cost"] += cost.get("cost", 0)
                    tournament_cost["input_tokens"] += cost.get("input_tokens", 0)
                    tournament_cost["output_tokens"] += cost.get("output_tokens", 0)
                    tournament_cost["time"] += cost.get("time", 0)

            # Summary step
            self._add_history(
                step="solution_tournament_summary",
                timestep=2,
                conversation=[],
                bracket=bracket,
                winning_solution_idx=winner_idx,
                winning_solution=winning_solution,
                total_cost=tournament_cost,
            )
            self._save_checkpoint()
            logger.info(f"[{self.bi}] Solution tournament complete. Winner: Solution {winner_idx}. Cost: ${tournament_cost['cost']:.4f}")

        # Build final conversation (problem + winning solution)
        final_convo = [
            {"role": "user", "content": self.default_prompt_template.format(problem=stmt)},
            {"role": "assistant", "content": winning_solution}
        ]

        return self._end_run(final_convo)
