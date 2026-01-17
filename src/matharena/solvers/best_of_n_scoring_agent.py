"""Best of N Scoring Agent: Generates N solutions and scores each individually to select best."""

import copy
import json
import random
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from hashlib import md5
from typing import Any, override

from jinja2 import Template
from loguru import logger

from matharena.api_client import APIClient
from matharena.solvers import BaseAgent, SolverResponse


class BestOfNScoringAgent(BaseAgent):
    """
    An agent that generates N candidate solutions for a math problem,
    scores each individually using an LLM judge, and returns the highest-scoring one.

    More token-efficient than tournament selection (N scores vs N-1 pairwise comparisons).
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
        self.RUN_ID = f"best_of_n_scoring_{self.model_config['model'].replace('/', '--')}_{self.problem_idx}_{parameter_hash}"

        # Configuration from scaffold (n_samples can be overridden in model config)
        self.n_samples = solver_config.get("n_samples", self.scaffold_config.get("n_samples", 8))
        self.generation_pool_size = self.scaffold_config.get("generation_pool_size", 4)
        self.scoring_pool_size = self.scaffold_config.get("scoring_pool_size", 4)

        # Load critic prompt from scaffold config
        critic_prompt = self.scaffold_config.get("prompts", {}).get("critic", None)
        if critic_prompt is None:
            raise ValueError("Critic prompt not found in scaffold config under 'prompts.critic'")
        self.critic_template = Template(critic_prompt)

        # Create API client
        simple_client_args = copy.deepcopy(default_api_client_args)
        for key in ["human_readable_id", "date", "other_params"]:
            simple_client_args.pop(key, None)
        self.client = APIClient(**simple_client_args)

        # Store the prompt template
        self.generation_prompt = default_prompt_template

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

    def _generate_one_solution(self) -> tuple[str, list[dict], dict]:
        """Generate a single solution. Returns (solution, conversation, cost_dict)."""
        prompt = self.generation_prompt.format(problem=self.stmt)
        convo = [{"role": "user", "content": prompt}]
        convo, call_cost = self._query_with_cost(self.client, convo)
        solution = convo[-1]["content"]
        return solution, convo, call_cost

    def _generate_solutions(self) -> tuple[list[str], list[list[dict]], list[dict]]:
        """Generate N solutions in parallel. Returns (solutions, convos, costs)."""
        solutions = []
        convos = []
        costs = []

        with ThreadPoolExecutor(max_workers=self.generation_pool_size) as executor:
            futures = [executor.submit(self._generate_one_solution) for _ in range(self.n_samples)]
            for future in as_completed(futures):
                solution, convo, call_cost = future.result()
                solutions.append(solution)
                convos.append(convo)
                costs.append(call_cost)

        return solutions, convos, costs

    def _parse_score(self, response: str) -> tuple[int, str]:
        """
        Parse score response.
        Expected format: "Score: X/10" at the end.
        Returns: (score, justification)
        """
        # Look for "Score: X/10" pattern
        match = re.search(r'Score:\s*(\d+)\s*/\s*10', response, re.IGNORECASE)
        if match:
            score = int(match.group(1))
            # Extract justification (everything before "Score:")
            justification_match = re.search(r'^(.*?)Score:', response, re.DOTALL | re.IGNORECASE)
            justification = justification_match.group(1).strip() if justification_match else ""
            return (max(0, min(10, score)), justification)

        # Fallback: look for just a number after "Score:"
        match = re.search(r'Score:\s*(\d+)', response, re.IGNORECASE)
        if match:
            score = int(match.group(1))
            return (max(0, min(10, score)), "Extracted score only")

        logger.warning(f"[{self.bi}] Failed to parse score from: {response[-200:]}")
        return (0, "Parse error")

    def _score_one_solution(self, solution: str, idx: int) -> tuple[int, str, list[dict], dict]:
        """Score a single solution. Returns (score, reason, convo, cost_dict)."""
        prompt = self.critic_template.render(
            problem=self.stmt,
            solution=solution
        )
        convo = [{"role": "user", "content": prompt}]
        convo, call_cost = self._query_with_cost(self.client, convo)

        response = convo[-1]["content"]
        score, reason = self._parse_score(response)

        return score, reason, convo, call_cost

    def _score_all_solutions(self, solutions: list[str]) -> tuple[list[int], list[str], list[list[dict]], list[dict]]:
        """Score all solutions in parallel. Returns (scores, reasons, convos, costs)."""
        results = [None] * len(solutions)

        with ThreadPoolExecutor(max_workers=self.scoring_pool_size) as executor:
            future_to_idx = {
                executor.submit(self._score_one_solution, sol, idx): idx
                for idx, sol in enumerate(solutions)
            }
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                score, reason, convo, call_cost = future.result()
                results[idx] = (score, reason, convo, call_cost)

        scores = [r[0] for r in results]
        reasons = [r[1] for r in results]
        convos = [r[2] for r in results]
        costs = [r[3] for r in results]

        return scores, reasons, convos, costs

    def _select_best(self, scores: list[int]) -> int:
        """Return index of highest-scoring solution (random tiebreak)."""
        max_score = max(scores)
        best_indices = [i for i, s in enumerate(scores) if s == max_score]
        return random.choice(best_indices)

    @override
    def solve(self, stmt: str) -> SolverResponse:
        """
        Main solve method:
        1. Generate N candidate solutions in parallel
        2. Score each solution individually
        3. Return the highest-scoring one
        """
        self._start_run(stmt)
        self._load_checkpoint_if_exists()

        # Step 1: Generate solutions
        if self._history_has_step("generation_summary"):
            logger.debug(f"[{self.bi}] Loading solutions from checkpoint.")
            gen_summary = self.get_history_step("generation_summary")
            solutions = gen_summary["solutions"]
        else:
            logger.debug(f"[{self.bi}] Generating {self.n_samples} candidate solutions.")
            solutions, gen_convos, gen_costs = self._generate_solutions()

            for i, (convo, cost) in enumerate(zip(gen_convos, gen_costs)):
                self._add_history(
                    step=f"generation_{i}",
                    timestep=1,
                    conversation=convo,
                    solution_index=i,
                    cost=cost,
                )

            total_gen_cost = {
                "cost": sum(c["cost"] for c in gen_costs),
                "input_tokens": sum(c["input_tokens"] for c in gen_costs),
                "output_tokens": sum(c["output_tokens"] for c in gen_costs),
                "time": sum(c["time"] for c in gen_costs),
            }

            self._add_history(
                step="generation_summary",
                timestep=1,
                conversation=[],
                n_solutions=len(solutions),
                solutions=solutions,
                total_cost=total_gen_cost,
            )
            self._save_checkpoint()
            logger.debug(f"[{self.bi}] Generated {len(solutions)} solutions. Cost: ${total_gen_cost['cost']:.4f}")

        # Step 2: Score solutions
        if self._history_has_step("scoring_summary"):
            logger.debug(f"[{self.bi}] Loading scores from checkpoint.")
            score_summary = self.get_history_step("scoring_summary")
            scores = score_summary["scores"]
            reasons = score_summary["reasons"]
        else:
            logger.debug(f"[{self.bi}] Scoring {len(solutions)} solutions.")
            scores, reasons, score_convos, score_costs = self._score_all_solutions(solutions)

            for i, (score, reason, convo, cost) in enumerate(zip(scores, reasons, score_convos, score_costs)):
                self._add_history(
                    step=f"scoring_{i}",
                    timestep=2,
                    conversation=convo,
                    solution_index=i,
                    score=score,
                    reason=reason,
                    cost=cost,
                )

            total_score_cost = {
                "cost": sum(c["cost"] for c in score_costs),
                "input_tokens": sum(c["input_tokens"] for c in score_costs),
                "output_tokens": sum(c["output_tokens"] for c in score_costs),
                "time": sum(c["time"] for c in score_costs),
            }

            self._add_history(
                step="scoring_summary",
                timestep=2,
                conversation=[],
                scores=scores,
                reasons=reasons,
                total_cost=total_score_cost,
            )
            self._save_checkpoint()
            logger.debug(f"[{self.bi}] Scoring complete. Scores: {scores}. Cost: ${total_score_cost['cost']:.4f}")

        # Step 3: Select best
        best_idx = self._select_best(scores)
        best_solution = solutions[best_idx]

        self._add_history(
            step="selection",
            timestep=3,
            conversation=[],
            best_index=best_idx,
            best_score=scores[best_idx],
            all_scores=scores,
        )

        logger.info(f"[{self.bi}] Selected solution {best_idx} with score {scores[best_idx]}/10")

        final_convo = [
            {"role": "user", "content": self.generation_prompt.format(problem=stmt)},
            {"role": "assistant", "content": best_solution}
        ]

        return self._end_run(final_convo)
