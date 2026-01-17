"""This module defines the base Agent class for solving math problems."""

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, override

from loguru import logger
from tqdm import tqdm

from matharena.api_client import APIClient
from matharena.solvers import (
    BaseSolver,
    SelfcheckAgent,
    SolverResponse,
    DeepSeekMathAgent,
    BestOfNAgent,
    PlanTournamentAgent,
    ConditionedSolverAgent,
    PlanGenerationAgent,
    PlanScoringAgent,
    PlanConditionedResponseAgent,
    ResponseScoringAgent,
    SequentialPlanRefinementAgent,
    # TODO: Uncomment when agent is implemented (Issue #2)
    # BranchingPlanExplorationAgent,
)


class AgentPool(BaseSolver):
    """
    A solver that manages a pool of agents to solve problems.
    """

    # Agent Registration:
    # To add a new agent:
    # 1. Create the agent class in src/matharena/solvers/{agent_name}.py
    # 2. Add import in src/matharena/solvers/__init__.py
    # 3. Add import above (line 11-26)
    # 4. Register here with scaffold_name as key (must match configs/agent_scaffolds/{name}.yaml)
    # 5. Create scaffold config in configs/agent_scaffolds/{name}.yaml with matching scaffold_name
    # 6. Create model config in configs/models/{provider}/{model}--{scaffold}.yaml
    AGENT_CLASSES = {
        "selfcheck": SelfcheckAgent,
        "deepseek_agent": DeepSeekMathAgent,
        "best_of_n": BestOfNAgent,
        "plan_tournament": PlanTournamentAgent,
        "conditioned_solver": ConditionedSolverAgent,
        "plan_generation": PlanGenerationAgent,
        "plan_scoring": PlanScoringAgent,
        "plan_conditioned_response": PlanConditionedResponseAgent,
        "response_scoring": ResponseScoringAgent,
        "sequential_plan_refinement": SequentialPlanRefinementAgent,
        # TODO: Uncomment when agent is implemented (Issue #2)
        # "branching_plan_exploration": BranchingPlanExplorationAgent,
    }

    def __init__(self, solver_config, default_prompt_template, default_api_client_args, last_chance_prompt):
        super().__init__(solver_config, default_prompt_template, default_api_client_args, last_chance_prompt)

        # AgentPool handles multithreading, individual agents use
        # model_config["concurrent_requests"] internally, but usually send 1 query at a time to APIClient.
        self.scaffold_config = self.solver_config["scaffold_config"]
        self.n_threads = self.scaffold_config.get("n_threads", 1)
        self.AGENT_CLASS = AgentPool.AGENT_CLASSES[self.scaffold_config["scaffold_name"]]

    def _run_agent(self, batch_idx: int, problem_idx: int, run_idx: int, stmt: tuple[str, Any]):
        agent = self.AGENT_CLASS(
            batch_idx=batch_idx,
            problem_idx=problem_idx,
            run_idx=run_idx,
            solver_config=self.solver_config,
            default_prompt_template=self.default_prompt_template,
            default_api_client_args=self.default_api_client_args,
        )
        # TODO: implement image support for agents if needed
        return agent.solve(stmt[0])

    @override
    def solve_batch(self, stmt_batch: list[tuple[str, Any]], batch_idx_to_problem_idx: dict[int, int], batch_idx_to_run_idx: dict[int, int]):
        """
        Solves a batch of problems. Handles multithreading, launching one Agent per problem.

        Args:
            stmt_batch (list[tuple[str, Any]]): A batch of problem statements as (text, image) pairs.
            batch_idx_to_problem_idx (dict[int, int]): A mapping from batch indices to original problem indices.
            batch_idx_to_run_idx (dict[int, int]): A mapping from batch indices to run indices.

        Yields:
            solver_response: A SolverResponse object containing the batch_index, the conversation array, detailed cost, and history for each problem.
        """
        logger.info(f"Starting agents with n_threads={self.n_threads} for batch of size {len(stmt_batch)}.")
        with ThreadPoolExecutor(max_workers=self.n_threads) as executor:
            futures = []
            for batch_idx, stmt in enumerate(stmt_batch):
                problem_idx = batch_idx_to_problem_idx[batch_idx]
                run_idx = batch_idx_to_run_idx[batch_idx]
                futures.append(executor.submit(self._run_agent, batch_idx, problem_idx, run_idx, stmt))
            iterator = as_completed(futures)
            iterator = tqdm(iterator, total=len(futures))
            for future in iterator:
                try:
                    solver_response = future.result()
                    logger.info(f"[{solver_response.idx}] Agent completed solving problem.")
                    yield solver_response
                except Exception as e:
                    logger.opt(exception=True).error(f"Agent failed with exception: {e}. Continuing with other runs.")
                    continue

    @override
    def last_chance(self, previous_response: SolverResponse) -> SolverResponse:
        """
        If the parser did not find the solution for some problem, the solver has a last chance to modify its response.
        TODO: see if we want last chance for agents, for now it's skipped as agents have plenty of chances to
        internally fix their mistakes and make sure they provided an answer.

        Args:
            previous_response (SolverResponse): The response this solver previously returned for a problem.

        Returns:
            SolverResponse: The modified response after reprompting the model to report an answer.
        """
        # last_chance_query = [{"role": "user", "content": self.last_chance_prompt}]
        logger.info(
            "AgentPool.last_chance called, but currently not implemented for agents. " "Returning the exact same thing."
        )
        return SolverResponse(
            idx=0,
            conversation=previous_response.conversation,
            detailed_cost=previous_response.detailed_cost,
            history=previous_response.history,
        )
