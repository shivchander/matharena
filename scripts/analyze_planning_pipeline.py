#!/usr/bin/env python3
"""
Analyze the planning pipeline: plan generation + scoring + conditioned response.

Computes:
- Total cost across all stages
- Accuracy metrics (pass@1, pass@N)
- Per-problem and aggregate statistics

Usage:
    python scripts/analyze_planning_pipeline.py \
        --comp apex/shortlist_2025 \
        --plan-gen openai/gpt-52--plan-gen \
        --plan-score openai/gpt-52--plan-scoring \
        --cond-resp openai/gpt-52--cond-resp

    # Simulate using only 16 plans (uses exact per-plan costs)
    python scripts/analyze_planning_pipeline.py \
        --comp apex/shortlist_2025 \
        --plan-gen openai/gpt-52--plan-gen \
        --plan-score openai/gpt-52--plan-scoring \
        --cond-resp openai/gpt-52--cond-resp-k16 \
        --sample-k 16
"""

import argparse
import json
import os
from pathlib import Path
from collections import defaultdict

import numpy as np


def load_outputs(comp: str, model: str) -> dict:
    """Load all output files for a model."""
    output_dir = Path("outputs") / comp / model
    outputs = {}

    if not output_dir.exists():
        print(f"Warning: {output_dir} does not exist")
        return outputs

    for filename in os.listdir(output_dir):
        if filename.endswith(".json"):
            problem_idx = int(filename.replace(".json", ""))
            with open(output_dir / filename) as f:
                outputs[problem_idx] = json.load(f)

    return outputs


def extract_cost(output: dict) -> dict:
    """Extract cost from an output dict."""
    cost = output.get("cost", {})
    return {
        "cost": cost.get("cost", 0) or 0,
        "input_tokens": cost.get("input_tokens", 0) or 0,
        "output_tokens": cost.get("output_tokens", 0) or 0,
        "reasoning_tokens": cost.get("reasoning_tokens", 0) or 0,
        "time": cost.get("time", 0) or 0,
    }


def extract_per_plan_costs(plan_gen_output: dict, plan_score_output: dict) -> tuple[list[dict], list[dict]]:
    """Extract per-plan costs from plan generation and scoring outputs.

    Returns (plan_gen_costs, plan_score_costs) where each is a list of cost dicts per plan.
    """
    # Plan generation: each plan is in a separate run (history[i])
    plan_gen_costs = []
    history = plan_gen_output.get("history", [])
    for run_history in history:
        run_cost = {"cost": 0, "input_tokens": 0, "output_tokens": 0, "reasoning_tokens": 0, "time": 0}
        for step in run_history:
            if step.get("step") == "plan_generated" and "cost" in step:
                c = step["cost"]
                run_cost = {
                    "cost": c.get("cost", 0) or 0,
                    "input_tokens": c.get("input_tokens", 0) or 0,
                    "output_tokens": c.get("output_tokens", 0) or 0,
                    "reasoning_tokens": c.get("reasoning_tokens", 0) or 0,
                    "time": c.get("time", 0) or 0,
                }
                break
        plan_gen_costs.append(run_cost)

    # Plan scoring: all scoring steps are in history[0] as scoring_0, scoring_1, etc.
    plan_score_costs = []
    if history := plan_score_output.get("history", []):
        if history and history[0]:
            # Build a dict of scoring costs by index
            scoring_by_idx = {}
            for step in history[0]:
                step_name = step.get("step", "")
                if step_name.startswith("scoring_") and "cost" in step:
                    try:
                        idx = int(step_name.replace("scoring_", ""))
                        c = step["cost"]
                        scoring_by_idx[idx] = {
                            "cost": c.get("cost", 0) or 0,
                            "input_tokens": c.get("input_tokens", 0) or 0,
                            "output_tokens": c.get("output_tokens", 0) or 0,
                            "reasoning_tokens": c.get("reasoning_tokens", 0) or 0,
                            "time": c.get("time", 0) or 0,
                        }
                    except ValueError:
                        pass

            # Convert to list ordered by index
            max_idx = max(scoring_by_idx.keys()) if scoring_by_idx else -1
            for i in range(max_idx + 1):
                plan_score_costs.append(scoring_by_idx.get(i, {"cost": 0, "input_tokens": 0, "output_tokens": 0, "reasoning_tokens": 0, "time": 0}))

    return plan_gen_costs, plan_score_costs


def get_sampled_indices(cond_resp_output: dict) -> list[int] | None:
    """Get sampled plan indices from conditioned response output, if any."""
    history = cond_resp_output.get("history", [])
    if history and history[0]:
        for step in history[0]:
            if step.get("step") == "plan_selected":
                return step.get("sampled_plan_indices")
    return None


def sum_costs(costs: list[dict]) -> dict:
    """Sum a list of cost dicts."""
    total = {"cost": 0, "input_tokens": 0, "output_tokens": 0, "reasoning_tokens": 0, "time": 0}
    for c in costs:
        for key in total:
            total[key] += c.get(key, 0) or 0
    return total


def sum_costs_for_indices(costs: list[dict], indices: list[int]) -> dict:
    """Sum costs for specific indices only."""
    total = {"cost": 0, "input_tokens": 0, "output_tokens": 0, "reasoning_tokens": 0, "time": 0}
    for idx in indices:
        if 0 <= idx < len(costs):
            for key in total:
                total[key] += costs[idx].get(key, 0) or 0
    return total


def compute_pass_at_k(correct: list, k: int, n_samples: int = 1000, seed: int = 42) -> float:
    """Compute pass@k using random sampling."""
    if not correct or k > len(correct):
        return 0.0

    n = len(correct)
    if k >= n:
        return 1.0 if any(correct) else 0.0

    rng = np.random.default_rng(seed)
    successes = 0

    for _ in range(n_samples):
        sample_indices = rng.choice(n, size=k, replace=False)
        if any(correct[i] for i in sample_indices):
            successes += 1

    return successes / n_samples


def compute_pass_at_k_with_variance(all_correct: list, k: int, n_seeds: int = 3, base_seed: int = 42) -> tuple:
    """Compute pass@k with multiple seeds for variance estimation.

    Returns (mean, std) across seeds.
    """
    seed_results = []

    for seed_idx in range(n_seeds):
        seed = base_seed + seed_idx * 1000
        pass_rates = []
        for correct in all_correct:
            pass_rates.append(compute_pass_at_k(correct, k, seed=seed))
        seed_results.append(np.mean(pass_rates))

    return np.mean(seed_results), np.std(seed_results)


def analyze_pipeline(
    comp: str,
    plan_gen_model: str,
    plan_score_model: str,
    cond_resp_model: str,
    verbose: bool = False,
    sample_k: int = None,
) -> dict:
    """Analyze the full planning pipeline.

    Args:
        sample_k: If provided, compute exact costs for only K plans by summing
                  per-plan costs. Uses sampled_plan_indices from the conditioned
                  response if available, otherwise uses first K plans.
    """

    # Load outputs from each stage
    plan_gen_outputs = load_outputs(comp, plan_gen_model)
    plan_score_outputs = load_outputs(comp, plan_score_model)
    cond_resp_outputs = load_outputs(comp, cond_resp_model)

    print(f"Loaded outputs:")
    print(f"  Plan generation: {len(plan_gen_outputs)} problems")
    print(f"  Plan scoring: {len(plan_score_outputs)} problems")
    print(f"  Conditioned response: {len(cond_resp_outputs)} problems")
    if sample_k:
        print(f"  Simulating {sample_k} plans (scaling upstream costs)")
    print()

    # Get common problems
    all_problems = set(plan_gen_outputs.keys()) & set(plan_score_outputs.keys()) & set(cond_resp_outputs.keys())
    print(f"Common problems: {len(all_problems)}")
    print()

    # Aggregate stats
    total_cost = {"cost": 0, "input_tokens": 0, "output_tokens": 0, "reasoning_tokens": 0, "time": 0}
    stage_costs = {
        "plan_gen": {"cost": 0, "input_tokens": 0, "output_tokens": 0, "reasoning_tokens": 0, "time": 0},
        "plan_score": {"cost": 0, "input_tokens": 0, "output_tokens": 0, "reasoning_tokens": 0, "time": 0},
        "cond_resp": {"cost": 0, "input_tokens": 0, "output_tokens": 0, "reasoning_tokens": 0, "time": 0},
    }

    all_correct = []
    problem_results = []

    for problem_idx in sorted(all_problems):
        plan_gen = plan_gen_outputs[problem_idx]
        plan_score = plan_score_outputs[problem_idx]
        cond_resp = cond_resp_outputs[problem_idx]

        # Extract per-plan costs for exact computation
        plan_gen_costs, plan_score_costs = extract_per_plan_costs(plan_gen, plan_score)

        # Get sampled indices from conditioned response (if using selection_sample_k)
        sampled_indices = get_sampled_indices(cond_resp)

        # Determine which indices to use for cost calculation
        actual_n_plans = plan_gen.get("N", 0)
        if sample_k and sample_k < actual_n_plans:
            # Use sample_k random indices (simulate having generated only K plans)
            # If sampled_indices exists, use those; otherwise use first K
            if sampled_indices and len(sampled_indices) == sample_k:
                indices_to_use = sampled_indices
            else:
                indices_to_use = list(range(sample_k))
            pg_cost = sum_costs_for_indices(plan_gen_costs, indices_to_use)
            ps_cost = sum_costs_for_indices(plan_score_costs, indices_to_use)
        else:
            # Use all plans
            pg_cost = extract_cost(plan_gen)
            ps_cost = extract_cost(plan_score)

        cr_cost = extract_cost(cond_resp)

        # Accumulate stage costs
        for key in total_cost:
            stage_costs["plan_gen"][key] += pg_cost[key]
            stage_costs["plan_score"][key] += ps_cost[key]
            stage_costs["cond_resp"][key] += cr_cost[key]
            total_cost[key] += pg_cost[key] + ps_cost[key] + cr_cost[key]

        # Extract correctness from conditioned response
        correct = cond_resp.get("correct", [])
        all_correct.append(correct)

        problem_total_cost = pg_cost["cost"] + ps_cost["cost"] + cr_cost["cost"]

        problem_results.append({
            "problem_idx": problem_idx,
            "n_plans": plan_gen.get("N", 0),
            "n_responses": cond_resp.get("N", 0),
            "correct": correct,
            "n_correct": sum(correct) if correct else 0,
            "pass_at_1": correct[0] if correct else False,
            "total_cost": problem_total_cost,
        })

        if verbose:
            print(f"Problem {problem_idx}: {sum(correct)}/{len(correct)} correct, cost=${problem_total_cost:.4f}")

    # Compute aggregate metrics
    n_problems = len(problem_results)
    n_responses_per_problem = problem_results[0]["n_responses"] if problem_results else 0

    # Flatten all correct
    flat_correct = [c for problem in all_correct for c in problem]

    # Per-problem pass@k
    pass_at_1_per_problem = [1 if any(c) else 0 for c in all_correct]

    # Compute pass@k for different k values with variance
    pass_at_k = {}
    for k in [1, 2, 4, 8, 16, 32, 64]:
        if k <= n_responses_per_problem:
            mean, std = compute_pass_at_k_with_variance(all_correct, k, n_seeds=3)
            pass_at_k[k] = {"mean": mean, "std": std}

    actual_n_plans = problem_results[0]["n_plans"] if problem_results else 0
    effective_n_plans = sample_k if sample_k and sample_k < actual_n_plans else actual_n_plans

    results = {
        "n_problems": n_problems,
        "n_plans_per_problem": actual_n_plans,
        "effective_n_plans": effective_n_plans,
        "sample_k": sample_k,
        "n_responses_per_problem": n_responses_per_problem,
        "total_cost": total_cost,
        "stage_costs": stage_costs,
        "cost_per_problem": total_cost["cost"] / n_problems if n_problems else 0,
        "pass_at_k": pass_at_k,
        "problems_with_correct": sum(pass_at_1_per_problem),
        "accuracy": sum(pass_at_1_per_problem) / n_problems if n_problems else 0,
        "total_correct": sum(flat_correct),
        "total_responses": len(flat_correct),
        "problem_results": problem_results,
    }

    return results


def print_results(results: dict):
    """Print formatted results."""
    print("=" * 60)
    print("PLANNING PIPELINE ANALYSIS")
    print("=" * 60)
    print()

    print(f"Problems analyzed: {results['n_problems']}")
    if results.get('sample_k') and results['sample_k'] < results['n_plans_per_problem']:
        print(f"Plans per problem: {results['n_plans_per_problem']} (simulating {results['effective_n_plans']} plans)")
    else:
        print(f"Plans per problem: {results['n_plans_per_problem']}")
    print(f"Responses per problem: {results['n_responses_per_problem']}")
    print()

    print("COSTS BY STAGE:")
    print("-" * 90)
    print(f"  {'Stage':<15} {'Cost':>10}  {'Output Tok':>14}  {'Reasoning Tok':>14}  {'Input Tok':>14}")
    print("-" * 90)
    for stage, cost in results["stage_costs"].items():
        print(f"  {stage:<15} ${cost['cost']:>8.4f}  {cost['output_tokens']:>14,}  {cost['reasoning_tokens']:>14,}  {cost['input_tokens']:>14,}")
    print("-" * 90)
    total = results["total_cost"]
    print(f"  {'TOTAL':<15} ${total['cost']:>8.4f}  {total['output_tokens']:>14,}  {total['reasoning_tokens']:>14,}  {total['input_tokens']:>14,}")
    print()
    print(f"Cost per problem: ${results['cost_per_problem']:.4f}")
    print()

    print("ACCURACY:")
    print("-" * 40)
    print(f"  Problems with ≥1 correct: {results['problems_with_correct']}/{results['n_problems']} ({results['accuracy']*100:.1f}%)")
    print(f"  Total correct responses: {results['total_correct']}/{results['total_responses']}")
    print()

    print("PASS@K (per problem, averaged over 3 seeds):")
    for k, stats in results["pass_at_k"].items():
        mean = stats["mean"]
        std = stats["std"]
        print(f"  pass@{k}: {mean*100:.2f}% ± {std*100:.2f}%")
    print()


def main():
    parser = argparse.ArgumentParser(description="Analyze planning pipeline")
    parser.add_argument("--comp", required=True, help="Competition path")
    parser.add_argument("--plan-gen", required=True, help="Plan generation model")
    parser.add_argument("--plan-score", required=True, help="Plan scoring model")
    parser.add_argument("--cond-resp", required=True, help="Conditioned response model")
    parser.add_argument("--sample-k", type=int, default=None,
                        help="Simulate using only K plans (uses exact per-plan costs from sampled indices)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show per-problem results")
    parser.add_argument("--output", "-o", help="Save results to JSON file")

    args = parser.parse_args()

    results = analyze_pipeline(
        args.comp,
        args.plan_gen,
        args.plan_score,
        args.cond_resp,
        verbose=args.verbose,
        sample_k=args.sample_k,
    )

    print_results(results)

    if args.output:
        # Remove problem_results for cleaner output
        output_results = {k: v for k, v in results.items() if k != "problem_results"}
        with open(args.output, "w") as f:
            json.dump(output_results, f, indent=2)
        print(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
