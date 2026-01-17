"""
Branching Plan Exploration Agent

Explores multiple reasoning paths through branching at decision points,
then synthesizes all leaf plans into one unified strategy.
"""

import copy
import re
import time
from concurrent.futures import ThreadPoolExecutor
from hashlib import md5
from typing import Any, Dict, List, Optional, Tuple, override

from jinja2 import Template
from loguru import logger

from matharena.api_client import APIClient
from matharena.solvers import BaseAgent, SolverResponse

class BranchingPlanExplorationAgent(BaseAgent):
    """
    Agent that builds a tree of reasoning plans by branching at decision points.
    
    Workflow:
    1. Generate initial plan
    2. If BRANCH_POINT detected, spawn parallel branches for each option
    3. Each branch continues planning (may branch further)
    4. Build tree up to max_branches depth
    5. Collect all leaf plans
    6. Synthesize into one unified plan
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
        self.RUN_ID = f"branch_plan_explore_{self.model_config['model'].replace('/', '--')}_{self.problem_idx}_{parameter_hash}"

        # Get agent-specific parameters
        self.max_branches = self.scaffold_config.get("max_branches", 4)
        self.branch_pool_size = self.scaffold_config.get("branch_pool_size", 8)

        # Get prompts from scaffold
        prompts = self.scaffold_config.get("prompts", {})

        initial_planner_prompt = prompts.get("initial_planner", None)
        if initial_planner_prompt is None:
            raise ValueError("Initial planner prompt not found in scaffold config under 'prompts.initial_planner'")
        self.initial_planner_template = Template(initial_planner_prompt)

        branch_continuation_prompt = prompts.get("branch_continuation", None)
        if branch_continuation_prompt is None:
            raise ValueError("Branch continuation prompt not found in scaffold config under 'prompts.branch_continuation'")
        self.branch_continuation_template = Template(branch_continuation_prompt)

        plan_synthesis_prompt = prompts.get("plan_synthesis", None)
        if plan_synthesis_prompt is None:
            raise ValueError("Plan synthesis prompt not found in scaffold config under 'prompts.plan_synthesis'")
        self.plan_synthesis_template = Template(plan_synthesis_prompt)

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

    @override
    def solve(self, stmt: str) -> SolverResponse:
        """
        Main solve method:
        1. Generate initial plan and explore branches
        2. Collect all leaf plans
        3. Synthesize into one unified plan
        4. Return final synthesized plan
        """
        self._start_run(stmt)
        self._load_checkpoint_if_exists()

        # Step 1: Generate initial plan and explore branches
        if self._history_has_step("tree_expanded"):
            logger.debug(f"[{self.bi}] Loading expanded tree from checkpoint.")
            tree = self.get_history_step("tree_expanded")["tree"]
        else:
            tree = self._explore_tree(stmt)

        # Step 2: Collect leaf plans
        if self._history_has_step("leaf_collection"):
            logger.debug(f"[{self.bi}] Loading leaf plans from checkpoint.")
            leaf_plans = self.get_history_step("leaf_collection")["leaf_plans"]
        else:
            leaf_plans = self._collect_leaf_plans(tree)
            self._add_history(
                step="leaf_collection",
                timestep=100,  # After tree expansion
                conversation=[],
                leaf_plans=leaf_plans,
                num_leaves=len(leaf_plans)
            )
            self._save_checkpoint()
            logger.debug(f"[{self.bi}] Collected {len(leaf_plans)} leaf plans.")

        # Step 3: Synthesize final plan
        if self._history_has_step("synthesis"):
            logger.debug(f"[{self.bi}] Loading synthesized plan from checkpoint.")
            final_plan = self.get_history_step("synthesis")["final_plan"]
        else:
            final_plan, synthesis_cost = self._synthesize_plans(stmt, leaf_plans)

            # Calculate total cost
            total_tree_cost = self._calculate_total_cost(tree)
            total_cost = {
                "tree_cost": total_tree_cost["cost"],
                "synthesis_cost": synthesis_cost["cost"],
                "total_cost": total_tree_cost["cost"] + synthesis_cost["cost"],
                "total_tokens": total_tree_cost["input_tokens"] + total_tree_cost["output_tokens"] + synthesis_cost["input_tokens"] + synthesis_cost["output_tokens"]
            }

            self._add_history(
                step="synthesis",
                timestep=101,
                conversation=[],
                final_plan=final_plan,
                synthesis_cost=synthesis_cost,
                total_cost=total_cost
            )
            self._save_checkpoint()
            logger.info(f"[{self.bi}] Synthesis complete. Total cost: ${total_cost['total_cost']:.4f}")

        # Build final conversation (user question + final plan)
        final_convo = [
            {"role": "user", "content": self.stmt},
            {"role": "assistant", "content": final_plan}
        ]

        return self._end_run(final_convo)

    def _explore_tree(self, problem: str) -> Dict[str, Any]:
        """
        Build the exploration tree by branching at decision points.

        Returns:
            Root node of the tree
        """
        # Check if we already have tree root in history
        if self._history_has_step("tree_root"):
            logger.debug(f"[{self.bi}] Loading tree root from checkpoint.")
            root = self.get_history_step("tree_root")["tree"]
        else:
            # Generate root node (initial plan)
            logger.debug(f"[{self.bi}] Generating initial plan (tree root).")
            root = self._generate_tree_node(problem, depth=0, path="root", chosen_option=None)

            # Add to history
            self._add_history(
                step="tree_root",
                timestep=1,
                conversation=root["conversation"],
                tree=root,
                cost=root["cost"]
            )
            self._save_checkpoint()
            logger.debug(f"[{self.bi}] Tree root generated. Cost: ${root['cost']['cost']:.4f}")

        # Recursively expand the tree
        self._expand_node(problem, root, current_depth=0)

        # Update history with fully expanded tree
        if not self._history_has_step("tree_expanded"):
            total_nodes = self._count_tree_nodes(root)
            self._add_history(
                step="tree_expanded",
                timestep=2,
                conversation=[],
                tree=root,
                total_nodes=total_nodes
            )
            self._save_checkpoint()
            logger.info(f"[{self.bi}] Tree expansion complete. Total nodes: {total_nodes}")

        return root

    def _generate_tree_node(self, problem: str, depth: int, path: str,
                            chosen_option: Optional[str]) -> Dict[str, Any]:
        """
        Generate a single tree node by querying the model.

        Args:
            problem: The original problem statement
            depth: Current depth in tree (0 = root)
            path: String describing path taken (e.g., "root -> Option 1")
            chosen_option: The option chosen to reach this node (None for root)

        Returns:
            Node dictionary with plan, branch_point, options, etc.
        """
        # Render appropriate prompt
        if depth == 0:
            # Root: use initial planner template
            prompt = self.initial_planner_template.render(problem=problem)
        else:
            # Branch continuation: use branch continuation template
            prompt = self.branch_continuation_template.render(
                problem=problem,
                current_path=path,
                chosen_option=chosen_option
            )

        # Query the model
        convo = [{"role": "user", "content": prompt}]
        convo, call_cost = self._query_with_cost(self.client, convo)

        response = convo[-1]["content"]

        # Parse for branch points
        parsed = self._parse_branch_point(response)

        if parsed:
            plan, branch_desc, options = parsed
        else:
            # No branch point - this is a leaf node
            plan = response
            branch_desc = None
            options = []

        # Build node
        node = {
            "plan": plan,
            "depth": depth,
            "path": path,
            "branch_point": branch_desc,
            "options": options,
            "children": [],
            "conversation": convo,
            "cost": call_cost,
            "is_leaf": len(options) == 0
        }

        logger.debug(f"[{self.bi}] Generated node at depth {depth}, path: {path}, "
                    f"branch_point: {branch_desc is not None}, options: {len(options)}")

        return node

    def _expand_node(self, problem: str, node: Dict[str, Any], current_depth: int):
        """
        Recursively expand a node by exploring its branches.

        Args:
            problem: The original problem
            node: The node to expand
            current_depth: Current depth in tree
        """
        # Base cases: don't expand if
        # 1. Already at max depth
        # 2. No branch points detected
        if current_depth >= self.max_branches or len(node["options"]) == 0:
            return

        # Check if we already expanded this node
        step_name = f"expand_{node['path'].replace(' ', '_').replace('->', '_')}"
        if self._history_has_step(step_name):
            logger.debug(f"[{self.bi}] Node {node['path']} already expanded, loading children.")
            # Children already exist in the node from checkpoint
            # Recursively expand each child
            for child in node["children"]:
                self._expand_node(problem, child, current_depth + 1)
            return

        logger.debug(f"[{self.bi}] Expanding node at depth {current_depth} with {len(node['options'])} options.")

        # Explore each option in parallel
        with ThreadPoolExecutor(max_workers=self.branch_pool_size) as executor:
            futures = []
            for option in node["options"]:
                child_path = f"{node['path']} -> {option}"
                future = executor.submit(
                    self._generate_tree_node,
                    problem,
                    current_depth + 1,
                    child_path,
                    option
                )
                futures.append(future)

            # Collect children
            for future in futures:
                child_node = future.result()
                node["children"].append(child_node)

        # Log expansion
        self._add_history(
            step=step_name,
            timestep=3 + current_depth,
            conversation=[],
            expanded_path=node["path"],
            num_children=len(node["children"])
        )
        self._save_checkpoint()
        logger.debug(f"[{self.bi}] Expanded {node['path']}: {len(node['children'])} children generated.")

        # Recursively expand each child
        for child in node["children"]:
            self._expand_node(problem, child, current_depth + 1)

    def _count_tree_nodes(self, node: Dict[str, Any]) -> int:
        """Count total nodes in tree (for logging)."""
        count = 1
        for child in node["children"]:
            count += self._count_tree_nodes(child)
        return count
    
    def _parse_branch_point(self, text: str) -> Optional[Tuple[str, str, List[str]]]:
        """
        Extract BRANCH_POINT from model output.

        Format:
            PLAN: [plan text]
            BRANCH_POINT: [description]
            OPTIONS: [Option 1 | Option 2 | Option 3]

        Returns:
            (plan, branch_point_description, options) if found, else None
        """
        # Extract PLAN
        plan_match = re.search(r'PLAN:\s*(.*?)(?=BRANCH_POINT:|$)', text, re.DOTALL | re.IGNORECASE)
        plan = plan_match.group(1).strip() if plan_match else text

        # Extract BRANCH_POINT
        branch_match = re.search(r'BRANCH_POINT:\s*(.*?)(?=OPTIONS:|$)', text, re.DOTALL | re.IGNORECASE)
        if not branch_match:
            # No branch point found
            return None

        branch_desc = branch_match.group(1).strip()

        # Extract OPTIONS
        options_match = re.search(r'OPTIONS:\s*(.*?)$', text, re.DOTALL | re.IGNORECASE)
        if not options_match:
            logger.warning(f"[{self.bi}] BRANCH_POINT found but no OPTIONS. Treating as leaf.")
            return None

        options_text = options_match.group(1).strip()

        # Split options by pipe (|) or numbered list
        # Try pipe-separated first
        if '|' in options_text:
            options = [opt.strip() for opt in options_text.split('|') if opt.strip()]
        else:
            # Try numbered list (1., 2., 3. or 1), 2), 3))
            numbered_options = re.findall(r'\d+[\.)]\s*(.*?)(?=\d+[\.)]|$)', options_text, re.DOTALL)
            if numbered_options:
                options = [opt.strip() for opt in numbered_options if opt.strip()]
            else:
                # Fallback: split by newlines
                options = [opt.strip() for opt in options_text.split('\n') if opt.strip()]

        # Validate we have at least 1 option
        if not options:
            logger.warning(f"[{self.bi}] BRANCH_POINT found but could not parse OPTIONS. Treating as leaf.")
            return None

        return plan, branch_desc, options
    
    def _collect_leaf_plans(self, tree: Dict[str, Any]) -> List[str]:
        """
        Recursively collect all leaf node plans from the tree.

        Args:
            tree: Tree structure from _explore_tree

        Returns:
            List of plan strings from all leaf nodes
        """
        leaves = []

        def collect_recursive(node):
            if node["is_leaf"] or len(node["children"]) == 0:
                # This is a leaf node
                leaves.append(node["plan"])
            else:
                # Recursively collect from children
                for child in node["children"]:
                    collect_recursive(child)

        collect_recursive(tree)
        return leaves
    
    def _synthesize_plans(self, problem: str, leaf_plans: List[str]) -> Tuple[str, dict]:
        """
        Use LLM to synthesize all leaf plans into one unified plan.

        Args:
            problem: Original problem statement
            leaf_plans: List of plans from leaf nodes

        Returns:
            (synthesized_plan, cost_dict)
        """
        # Handle edge case: only one leaf plan
        if len(leaf_plans) == 1:
            logger.debug(f"[{self.bi}] Only one leaf plan, returning it directly (no synthesis needed).")
            return leaf_plans[0], {"cost": 0, "input_tokens": 0, "output_tokens": 0, "time": 0}

        # Build synthesis prompt with all leaf plans
        # Create indexed list for template
        indexed_plans = [{"index": i, "plan": plan} for i, plan in enumerate(leaf_plans)]

        prompt = self.plan_synthesis_template.render(
            problem=problem,
            leaf_plans=indexed_plans,
            num_plans=len(leaf_plans)
        )

        # Query the model
        convo = [{"role": "user", "content": prompt}]
        convo, call_cost = self._query_with_cost(self.client, convo)

        synthesized_plan = convo[-1]["content"]

        logger.debug(f"[{self.bi}] Synthesized {len(leaf_plans)} plans. Cost: ${call_cost['cost']:.4f}")

        return synthesized_plan, call_cost
    
    def _calculate_total_cost(self, tree: Dict[str, Any]) -> dict:
        """
        Sum up costs from all nodes in the tree.

        Returns:
            Dictionary with total cost, input_tokens, output_tokens, time
        """
        total_cost = {"cost": 0, "input_tokens": 0, "output_tokens": 0, "time": 0}

        def sum_recursive(node):
            cost = node.get("cost", {})
            total_cost["cost"] += cost.get("cost", 0)
            total_cost["input_tokens"] += cost.get("input_tokens", 0)
            total_cost["output_tokens"] += cost.get("output_tokens", 0)
            total_cost["time"] += cost.get("time", 0)

            for child in node.get("children", []):
                sum_recursive(child)

        sum_recursive(tree)
        return total_cost
