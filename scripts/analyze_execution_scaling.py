#!/usr/bin/env python3
"""
Analyze execution scaling (direct best-of-N without planning).

Computes:
- Total cost
- Accuracy metrics (pass@1, pass@2, pass@4, ..., pass@N)
- Per-problem and aggregate statistics

Usage:
    python scripts/analyze_execution_scaling.py \
        --comp apex/shortlist_2025 \
        --model openai/gpt-52--none
"""

import argparse
import json
import os
from pathlib import Path

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
        "time": cost.get("time", 0) or 0,
    }


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


def analyze_execution(comp: str, model: str, verbose: bool = False) -> dict:
    """Analyze execution scaling (direct best-of-N)."""

    # Load outputs
    outputs = load_outputs(comp, model)
    print(f"Loaded {len(outputs)} problems from {model}")
    print()

    # Aggregate stats
    total_cost = {"cost": 0, "input_tokens": 0, "output_tokens": 0, "time": 0}
    all_correct = []
    problem_results = []

    for problem_idx in sorted(outputs.keys()):
        output = outputs[problem_idx]

        # Extract cost
        cost = extract_cost(output)
        for key in total_cost:
            total_cost[key] += cost[key]

        # Extract correctness
        correct = output.get("correct", [])
        all_correct.append(correct)

        problem_results.append({
            "problem_idx": problem_idx,
            "n_responses": output.get("N", 0),
            "correct": correct,
            "n_correct": sum(correct) if correct else 0,
            "pass_at_1": correct[0] if correct else False,
            "cost": cost["cost"],
        })

        if verbose:
            print(f"Problem {problem_idx}: {sum(correct)}/{len(correct)} correct, cost=${cost['cost']:.4f}")

    # Compute aggregate metrics
    n_problems = len(problem_results)
    n_responses_per_problem = problem_results[0]["n_responses"] if problem_results else 0

    # Per-problem pass@k
    pass_at_1_per_problem = [1 if any(c) else 0 for c in all_correct]

    # Compute pass@k for different k values with variance
    pass_at_k = {}
    for k in [1, 2, 4, 8, 16, 32, 64]:
        if k <= n_responses_per_problem:
            mean, std = compute_pass_at_k_with_variance(all_correct, k, n_seeds=3)
            pass_at_k[k] = {"mean": mean, "std": std}

    # Flatten all correct
    flat_correct = [c for problem in all_correct for c in problem]

    results = {
        "n_problems": n_problems,
        "n_responses_per_problem": n_responses_per_problem,
        "total_cost": total_cost,
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
    print("EXECUTION SCALING ANALYSIS (Direct Best-of-N)")
    print("=" * 60)
    print()

    print(f"Problems analyzed: {results['n_problems']}")
    print(f"Responses per problem: {results['n_responses_per_problem']}")
    print()

    print("COSTS:")
    print("-" * 40)
    total = results["total_cost"]
    print(f"  Total cost: ${total['cost']:.4f}")
    print(f"  Input tokens: {total['input_tokens']:,}")
    print(f"  Output tokens: {total['output_tokens']:,}")
    print(f"  Cost per problem: ${results['cost_per_problem']:.4f}")
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
    parser = argparse.ArgumentParser(description="Analyze execution scaling")
    parser.add_argument("--comp", required=True, help="Competition path")
    parser.add_argument("--model", required=True, help="Model path (e.g., openai/gpt-52--none)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show per-problem results")
    parser.add_argument("--output", "-o", help="Save results to JSON file")

    args = parser.parse_args()

    results = analyze_execution(args.comp, args.model, verbose=args.verbose)

    print_results(results)

    if args.output:
        # Remove problem_results for cleaner output
        output_results = {k: v for k, v in results.items() if k != "problem_results"}
        with open(args.output, "w") as f:
            json.dump(output_results, f, indent=2)
        print(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
