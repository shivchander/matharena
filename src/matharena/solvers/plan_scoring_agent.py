"""Plan Scoring Agent: Scores plans from upstream PlanGenerationAgent."""

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


class PlanScoringAgent(BaseAgent):
    """
    Agent that reads plans from an upstream PlanGenerationAgent's output,
    scores each individually using an LLM judge (Score: X/10 format),
    and identifies the best plan.

    Reads plans from messages[0..K-1] where K = number of runs from PlanGenerationAgent.
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
        self.RUN_ID = f"plan_score_{self.model_config['model'].replace('/', '--')}_{self.problem_idx}_{parameter_hash}"

        # Configuration
        self.scoring_pool_size = self.scaffold_config.get("scoring_pool_size", 4)

        # Temperature support
        self.temperature = solver_config.get(
            "temperature",
            self.scaffold_config.get("temperature", None)
        )

        # Plan source model (where to read plans from)
        self.plan_source_model = solver_config.get(
            "plan_source_model",
            self.scaffold_config.get("plan_source_model", None)
        )
        if self.plan_source_model is None:
            raise ValueError("plan_source_model must be specified in solver config or scaffold config")

        # Competition name (for building output path)
        self.competition = solver_config.get("competition", None)
        if self.competition is None:
            raise ValueError("competition must be specified in solver config")

        # Load plan critic prompt from scaffold config
        plan_critic_prompt = self.scaffold_config.get("prompts", {}).get("plan_critic", None)
        if plan_critic_prompt is None:
            raise ValueError("Plan critic prompt not found in scaffold config under 'prompts.plan_critic'")
        self.plan_critic_template = Template(plan_critic_prompt)

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

    def _load_plans(self) -> list[str]:
        """
        Load plans from upstream plan generation agent output.

        Supports two sources:
        1. PlanGenerationAgent: Plans in messages[0..K-1] (one plan per run)
        2. BatchPlanGenerationAgent: Plans in history[*] under 'plan_generated' steps
        """
        plan_output_path = os.path.join(
            "outputs",
            self.competition,
            self.plan_source_model,
            f"{self.problem_idx}.json"
        )

        if not os.path.exists(plan_output_path):
            raise FileNotFoundError(
                f"Plan source not found at {plan_output_path}. "
                f"Run {self.plan_source_model} first for problem {self.problem_idx}"
            )

        try:
            with open(plan_output_path, "r") as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(
                f"Corrupted JSON at {plan_output_path}: {e}. "
                f"The upstream plan generation output may be incomplete."
            )

        plans = []

        # First, check history for batch-generated plans (from BatchPlanGenerationAgent)
        history_list = data.get("history", [])
        for run_history in history_list:
            if run_history is None:
                continue
            for step in run_history:
                if step.get("step") == "plan_generated":
                    plan = step.get("plan", "")
                    if plan:
                        plans.append(plan)

        # If no plans found in history, fall back to messages (from PlanGenerationAgent)
        if not plans:
            messages = data.get("messages", [])
            for convo in messages:
                if convo and len(convo) >= 1:
                    # Find the assistant's response
                    for msg in reversed(convo):
                        if msg.get("role") == "assistant":
                            content = msg.get("content", "")
                            if content:
                                plans.append(content)
                            break

        if not plans:
            raise ValueError(f"No plans found in {plan_output_path}. Check that plan generation ran successfully.")

        return plans

    def _parse_score(self, response: str) -> tuple[int, str]:
        """
        Parse score response.
        Expected format: JSON with "reasoning" and "score" fields.
        Fallback: "Score: X/10" pattern for backward compatibility.
        Returns: (score, justification/reasoning)
        """
        # Try JSON parsing first
        # Look for JSON object in the response
        json_match = re.search(r'\{[^{}]*"score"\s*:\s*\d+[^{}]*\}', response, re.DOTALL)
        if json_match:
            try:
                data = json.loads(json_match.group(0))
                score = int(data.get("score", 0))
                reasoning = data.get("reasoning", "")
                return (max(0, min(10, score)), reasoning)
            except (json.JSONDecodeError, ValueError):
                pass

        # Fallback: Look for "Score: X/10" pattern
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

    def _score_one_plan(self, plan: str, idx: int) -> tuple[int, str, list[dict], dict]:
        """Score a single plan. Returns (score, justification, convo, cost_dict)."""
        prompt = self.plan_critic_template.render(
            problem=self.stmt,
            plan=plan
        )
        convo = [{"role": "user", "content": prompt}]
        convo, call_cost = self._query_with_cost(self.client, convo)

        response = convo[-1]["content"]
        score, justification = self._parse_score(response)

        return score, justification, convo, call_cost

    def _score_all_plans(self, plans: list[str]) -> tuple[list[int], list[str], list[list[dict]], list[dict]]:
        """Score all plans in parallel. Returns (scores, justifications, convos, costs)."""
        results = [None] * len(plans)

        with ThreadPoolExecutor(max_workers=self.scoring_pool_size) as executor:
            future_to_idx = {
                executor.submit(self._score_one_plan, plan, idx): idx
                for idx, plan in enumerate(plans)
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
        """Return index of highest-scoring plan (random tiebreak)."""
        max_score = max(scores)
        best_indices = [i for i, s in enumerate(scores) if s == max_score]
        return random.choice(best_indices)

    @override
    def solve(self, stmt: str) -> SolverResponse:
        """
        Main solve method:
        1. Load plans from upstream PlanGenerationAgent (from messages array)
        2. Score each plan individually
        3. Identify the best plan
        """
        self._start_run(stmt)
        self._load_checkpoint_if_exists()

        # Step 1: Load plans
        if self._history_has_step("plans_loaded"):
            logger.debug(f"[{self.bi}] Loading plans from checkpoint.")
            plans_loaded = self.get_history_step("plans_loaded")
            plans = plans_loaded["plans"]
        else:
            logger.debug(f"[{self.bi}] Loading plans from {self.plan_source_model}.")
            plans = self._load_plans()

            self._add_history(
                step="plans_loaded",
                timestep=1,
                conversation=[],
                plan_source_model=self.plan_source_model,
                n_plans=len(plans),
                plans=plans,
            )
            self._save_checkpoint()
            logger.debug(f"[{self.bi}] Loaded {len(plans)} plans from upstream.")

        # Step 2: Score plans
        if self._history_has_step("scoring_summary"):
            logger.debug(f"[{self.bi}] Loading scores from checkpoint.")
            score_summary = self.get_history_step("scoring_summary")
            scores = score_summary["scores"]
            best_idx = score_summary["best_plan_idx"]
            best_plan = score_summary["best_plan"]
        else:
            logger.debug(f"[{self.bi}] Scoring {len(plans)} plans.")
            scores, justifications, score_convos, score_costs = self._score_all_plans(plans)

            for i, (score, justification, convo, cost) in enumerate(zip(scores, justifications, score_convos, score_costs)):
                self._add_history(
                    step=f"scoring_{i}",
                    timestep=2,
                    conversation=convo,
                    plan_index=i,
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
            best_plan = plans[best_idx]

            self._add_history(
                step="scoring_summary",
                timestep=2,
                conversation=[],
                scores=scores,
                justifications=justifications,
                plans=plans,  # Store plans for downstream agents
                best_plan_idx=best_idx,
                best_plan=best_plan,
                best_score=scores[best_idx],
                total_cost=total_score_cost,
            )
            self._save_checkpoint()
            logger.debug(f"[{self.bi}] Scoring complete. Scores: {scores}. Cost: ${total_score_cost['cost']:.4f}")

        # Get best plan from summary
        score_summary = self.get_history_step("scoring_summary")
        best_idx = score_summary["best_plan_idx"]
        best_plan = score_summary["best_plan"]
        best_score = score_summary["best_score"]

        logger.info(f"[{self.bi}] Best plan is #{best_idx} with score {best_score}/10")

        # Build final conversation (problem + best plan)
        final_convo = [
            {"role": "user", "content": self.default_prompt_template.format(problem=stmt)},
            {"role": "assistant", "content": best_plan}
        ]

        return self._end_run(final_convo)
