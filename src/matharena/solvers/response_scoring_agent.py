"""Response Scoring Agent: Scores solutions from upstream sources."""

import copy
import json
import os
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


class ResponseScoringAgent(BaseAgent):
    """
    Agent that reads solutions from an upstream source (PureModelSolver or PlanConditionedResponseAgent),
    scores each individually using an LLM judge (Score: X/10 format),
    and identifies the best solution.

    Supports reading from both:
    - PureModelSolver output (solutions in messages[0..N-1])
    - Agent output (solutions in history[0]["generation_summary"]["solutions"])
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
        self.RUN_ID = f"resp_score_{self.model_config['model'].replace('/', '--')}_{self.problem_idx}_{parameter_hash}"

        # Configuration
        self.scoring_pool_size = self.scaffold_config.get("scoring_pool_size", 4)

        # Temperature support
        self.temperature = solver_config.get(
            "temperature",
            self.scaffold_config.get("temperature", None)
        )

        # Response source model (where to read solutions from)
        self.response_source_model = solver_config.get(
            "response_source_model",
            self.scaffold_config.get("response_source_model", None)
        )
        if self.response_source_model is None:
            raise ValueError("response_source_model must be specified in solver config or scaffold config")

        # Competition name (for building output path)
        self.competition = solver_config.get("competition", None)
        if self.competition is None:
            raise ValueError("competition must be specified in solver config")

        # Load solution critic prompt from scaffold config
        solution_critic_prompt = self.scaffold_config.get("prompts", {}).get("solution_critic", None)
        if solution_critic_prompt is None:
            raise ValueError("Solution critic prompt not found in scaffold config under 'prompts.solution_critic'")
        self.solution_critic_template = Template(solution_critic_prompt)

        # Create API client with temperature if specified
        simple_client_args = copy.deepcopy(default_api_client_args)
        for key in ["human_readable_id", "date", "other_params"]:
            simple_client_args.pop(key, None)

        if self.temperature is not None:
            simple_client_args["temperature"] = self.temperature

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

    def _load_solutions(self) -> list[str]:
        """
        Load solutions from upstream source.
        Supports both PureModelSolver (messages) and Agent (history) output formats.
        """
        response_output_path = os.path.join(
            "outputs",
            self.competition,
            self.response_source_model,
            f"{self.problem_idx}.json"
        )

        if not os.path.exists(response_output_path):
            raise FileNotFoundError(
                f"Response source not found at {response_output_path}. "
                f"Run {self.response_source_model} first for problem {self.problem_idx}"
            )

        try:
            with open(response_output_path, "r") as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(
                f"Corrupted JSON at {response_output_path}: {e}. "
                f"The upstream response output may be incomplete."
            )

        # Try 1: Agent format - look for "generation_summary" in history
        history = data.get("history", [])
        if history and isinstance(history[0], list) and history[0]:
            # Agent output: history is List[List[dict]]
            for step in history[0]:
                if step.get("step") == "generation_summary":
                    solutions = step.get("solutions", [])
                    if solutions:
                        logger.debug(f"[{self.bi}] Loaded {len(solutions)} solutions from agent history.")
                        return solutions

        # Try 2: PureModelSolver format - read from messages
        messages = data.get("messages", [])
        if messages:
            solutions = []
            for convo in messages:
                if convo and len(convo) >= 2:
                    # The assistant's response is the last message
                    for msg in reversed(convo):
                        if msg.get("role") == "assistant":
                            content = msg.get("content", "")
                            if content:
                                solutions.append(content)
                            break
            if solutions:
                logger.debug(f"[{self.bi}] Loaded {len(solutions)} solutions from PureModelSolver messages.")
                return solutions

        raise ValueError(f"No solutions found in {response_output_path}. Check that the source ran successfully.")

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
        """Score a single solution. Returns (score, justification, convo, cost_dict)."""
        prompt = self.solution_critic_template.render(
            problem=self.stmt,
            solution=solution
        )
        convo = [{"role": "user", "content": prompt}]
        convo, call_cost = self._query_with_cost(self.client, convo)

        response = convo[-1]["content"]
        score, justification = self._parse_score(response)

        return score, justification, convo, call_cost

    def _score_all_solutions(self, solutions: list[str]) -> tuple[list[int], list[str], list[list[dict]], list[dict]]:
        """Score all solutions in parallel. Returns (scores, justifications, convos, costs)."""
        results = [None] * len(solutions)

        with ThreadPoolExecutor(max_workers=self.scoring_pool_size) as executor:
            future_to_idx = {
                executor.submit(self._score_one_solution, sol, idx): idx
                for idx, sol in enumerate(solutions)
            }
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                score, justification, convo, call_cost = future.result()
                results[idx] = (score, justification, convo, call_cost)

        scores = [r[0] for r in results]
        justifications = [r[1] for r in results]
        convos = [r[2] for r in results]
        costs = [r[3] for r in results]

        return scores, justifications, convos, costs

    def _select_best(self, scores: list[int]) -> int:
        """Return index of highest-scoring solution (random tiebreak)."""
        max_score = max(scores)
        best_indices = [i for i, s in enumerate(scores) if s == max_score]
        return random.choice(best_indices)

    @override
    def solve(self, stmt: str) -> SolverResponse:
        """
        Main solve method:
        1. Load solutions from upstream source
        2. Score each solution individually
        3. Return the best solution
        """
        self._start_run(stmt)
        self._load_checkpoint_if_exists()

        # Step 1: Load solutions
        if self._history_has_step("solutions_loaded"):
            logger.debug(f"[{self.bi}] Loading solutions from checkpoint.")
            solutions_loaded = self.get_history_step("solutions_loaded")
            solutions = solutions_loaded["solutions"]
        else:
            logger.debug(f"[{self.bi}] Loading solutions from {self.response_source_model}.")
            solutions = self._load_solutions()

            self._add_history(
                step="solutions_loaded",
                timestep=1,
                conversation=[],
                response_source_model=self.response_source_model,
                n_solutions=len(solutions),
                solutions=solutions,
            )
            self._save_checkpoint()
            logger.debug(f"[{self.bi}] Loaded {len(solutions)} solutions from upstream.")

        # Step 2: Score solutions
        if self._history_has_step("scoring_summary"):
            logger.debug(f"[{self.bi}] Loading scores from checkpoint.")
            score_summary = self.get_history_step("scoring_summary")
            scores = score_summary["scores"]
            best_idx = score_summary["best_solution_idx"]
            best_solution = score_summary["best_solution"]
        else:
            logger.debug(f"[{self.bi}] Scoring {len(solutions)} solutions.")
            scores, justifications, score_convos, score_costs = self._score_all_solutions(solutions)

            for i, (score, justification, convo, cost) in enumerate(zip(scores, justifications, score_convos, score_costs)):
                self._add_history(
                    step=f"scoring_{i}",
                    timestep=2,
                    conversation=convo,
                    solution_index=i,
                    score=score,
                    justification=justification,
                    cost=cost,
                )

            total_score_cost = {
                "cost": sum(c["cost"] for c in score_costs),
                "input_tokens": sum(c["input_tokens"] for c in score_costs),
                "output_tokens": sum(c["output_tokens"] for c in score_costs),
                "time": sum(c["time"] for c in score_costs),
            }

            # Step 3: Select best
            best_idx = self._select_best(scores)
            best_solution = solutions[best_idx]

            self._add_history(
                step="scoring_summary",
                timestep=2,
                conversation=[],
                scores=scores,
                justifications=justifications,
                best_solution_idx=best_idx,
                best_solution=best_solution,
                best_score=scores[best_idx],
                total_cost=total_score_cost,
            )
            self._save_checkpoint()
            logger.debug(f"[{self.bi}] Scoring complete. Scores: {scores}. Cost: ${total_score_cost['cost']:.4f}")

        # Get best solution from summary
        score_summary = self.get_history_step("scoring_summary")
        best_idx = score_summary["best_solution_idx"]
        best_solution = score_summary["best_solution"]
        best_score = score_summary["best_score"]

        logger.info(f"[{self.bi}] Best solution is #{best_idx} with score {best_score}/10")

        # Build final conversation (problem + best solution)
        final_convo = [
            {"role": "user", "content": self.default_prompt_template.format(problem=stmt)},
            {"role": "assistant", "content": best_solution}
        ]

        return self._end_run(final_convo)
