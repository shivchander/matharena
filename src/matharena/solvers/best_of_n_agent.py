"""Best of N Agent: Generates N candidate solutions and uses LLM judge to select the best one."""

import copy
import json
import random
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


class BestOfNAgent(BaseAgent):
    """
    An agent that generates N candidate solutions for a math problem,
    scores each using an LLM judge with a critic prompt, and returns the best one.
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
        self.RUN_ID = f"best_of_n_{self.model_config['model'].replace('/', '--')}_{self.problem_idx}_{parameter_hash}"

        # Configuration from scaffold (n_samples can be overridden in model config)
        self.n_samples = self.scaffold_config.get("n_samples", 8)
        self.generation_pool_size = self.scaffold_config.get("generation_pool_size", 4)
        self.judging_pool_size = self.scaffold_config.get("judging_pool_size", 4)

        # Load critic prompt from scaffold config
        critic_prompt = self.scaffold_config.get("prompts", {}).get("critic", None)
        if critic_prompt is None:
            raise ValueError("Critic prompt not found in scaffold config under 'prompts.critic'")
        self.critic_template = Template(critic_prompt)

        # Create API client (same model for generation and judging)
        simple_client_args = copy.deepcopy(default_api_client_args)
        # Remove non-API client args
        for key in ["human_readable_id", "date", "other_params"]:
            simple_client_args.pop(key, None)
        self.client = APIClient(**simple_client_args)

        # Store the prompt template
        self.generation_prompt = default_prompt_template

    def _query_with_cost(self, client: APIClient, query: list[dict[str, Any]]) -> tuple[list[dict], dict]:
        """
        A wrapper that runs a query and returns both the conversation and the per-call cost.
        This duplicates some logic from _query to capture the actual per-call cost directly
        (avoiding race conditions when running in parallel).

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

        # Return the per-call cost (directly from API response, not diffed)
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

    def _parse_critic_response(self, response: str) -> tuple[int, str]:
        """
        Parse JSON response from critic.
        Expected format: {"reasoning": "...", "score": <0-10>}

        Returns: (score, reasoning)
        Falls back to score=0 on parse failure.
        """
        # Try direct JSON parse
        try:
            data = json.loads(response.strip())
            score = int(data.get("score", 0))
            reasoning = data.get("reasoning", "")
            return (max(0, min(10, score)), reasoning)
        except (json.JSONDecodeError, ValueError):
            pass

        # Try to extract JSON from response (model might add extra text)
        json_match = re.search(r'\{[^{}]*"score"\s*:\s*\d+[^{}]*\}', response)
        if json_match:
            try:
                data = json.loads(json_match.group())
                score = int(data.get("score", 0))
                reasoning = data.get("reasoning", "")
                return (max(0, min(10, score)), reasoning)
            except (json.JSONDecodeError, ValueError):
                pass

        # Fallback: try to find just a score
        score_match = re.search(r'"score"\s*:\s*(\d+)', response)
        if score_match:
            return (int(score_match.group(1)), "Parse error - extracted score only")

        logger.warning(f"[{self.bi}] Failed to parse critic response: {response[:200]}")
        return (0, "Parse error")

    def _judge_one_solution(self, solution: str) -> tuple[int, str, list[dict], dict]:
        """Score a single solution using the critic prompt. Returns (score, reasoning, convo, cost_dict)."""
        critic_prompt = self.critic_template.render(
            question=self.stmt,
            solution=solution
        )
        convo = [{"role": "user", "content": critic_prompt}]
        convo, call_cost = self._query_with_cost(self.client, convo)

        response = convo[-1]["content"]
        score, reasoning = self._parse_critic_response(response)
        return score, reasoning, convo, call_cost

    def _judge_all_solutions(self, solutions: list[str]) -> tuple[list[int], list[str], list[list[dict]], list[dict]]:
        """Score all solutions in parallel. Returns (scores, reasonings, convos, costs)."""
        results = [None] * len(solutions)

        with ThreadPoolExecutor(max_workers=self.judging_pool_size) as executor:
            future_to_idx = {
                executor.submit(self._judge_one_solution, sol): idx
                for idx, sol in enumerate(solutions)
            }
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                score, reasoning, convo, call_cost = future.result()
                results[idx] = (score, reasoning, convo, call_cost)

        scores = [r[0] for r in results]
        reasonings = [r[1] for r in results]
        convos = [r[2] for r in results]
        costs = [r[3] for r in results]

        return scores, reasonings, convos, costs

    def _select_best_index(self, scores: list[int]) -> int:
        """
        Return the index of the highest-scoring solution.
        Random tie-breaking among tied highest scores.
        """
        max_score = max(scores)
        best_indices = [i for i, s in enumerate(scores) if s == max_score]
        return random.choice(best_indices)

    @override
    def solve(self, stmt: str) -> SolverResponse:
        """
        Main solve method:
        1. Generate N candidate solutions in parallel
        2. Score each with the critic prompt in parallel
        3. Select the best one (random tie-breaking)
        4. Return it as the final response
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

            # Record each generation with its cost
            for i, (convo, cost) in enumerate(zip(gen_convos, gen_costs)):
                self._add_history(
                    step=f"generation_{i}",
                    timestep=1,
                    conversation=convo,
                    solution_index=i,
                    cost=cost,
                )

            # Calculate total generation cost
            total_gen_cost = {
                "cost": sum(c["cost"] for c in gen_costs),
                "input_tokens": sum(c["input_tokens"] for c in gen_costs),
                "output_tokens": sum(c["output_tokens"] for c in gen_costs),
                "time": sum(c["time"] for c in gen_costs),
            }

            # Summary step with total cost
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

        # Step 2: Judge solutions
        if self._history_has_step("judging_summary"):
            logger.debug(f"[{self.bi}] Loading scores from checkpoint.")
            judge_summary = self.get_history_step("judging_summary")
            scores = judge_summary["scores"]
            reasonings = judge_summary["reasonings"]
        else:
            logger.debug(f"[{self.bi}] Judging {len(solutions)} solutions.")
            scores, reasonings, judge_convos, judge_costs = self._judge_all_solutions(solutions)

            # Record each judging with its cost
            for i, (score, reasoning, convo, cost) in enumerate(zip(scores, reasonings, judge_convos, judge_costs)):
                self._add_history(
                    step=f"judging_{i}",
                    timestep=2,
                    conversation=convo,
                    solution_index=i,
                    score=score,
                    reasoning=reasoning,
                    cost=cost,
                )

            # Calculate total judging cost
            total_judge_cost = {
                "cost": sum(c["cost"] for c in judge_costs),
                "input_tokens": sum(c["input_tokens"] for c in judge_costs),
                "output_tokens": sum(c["output_tokens"] for c in judge_costs),
                "time": sum(c["time"] for c in judge_costs),
            }

            # Summary step with total cost
            self._add_history(
                step="judging_summary",
                timestep=2,
                conversation=[],
                scores=scores,
                reasonings=reasonings,
                total_cost=total_judge_cost,
            )
            self._save_checkpoint()
            logger.debug(f"[{self.bi}] Judging complete. Scores: {scores}. Cost: ${total_judge_cost['cost']:.4f}")

        # Step 3: Select best
        best_idx = self._select_best_index(scores)
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

        # Build final conversation (user question + best solution)
        final_convo = [
            {"role": "user", "content": self.generation_prompt.format(problem=stmt)},
            {"role": "assistant", "content": best_solution}
        ]

        return self._end_run(final_convo)
