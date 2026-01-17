#!/usr/bin/env python3
"""
Analyze pass@2^N metrics and costs for a model's outputs.

Computes accuracy at different budget levels (pass@1, pass@2, pass@4, ..., pass@64)
by randomly sampling subsets of runs with a fixed seed for reproducibility.

Usage:
    uv run python scripts/analyze_budget_accuracy.py --model openai/gpt-52--none --comp apex/shortlist_2025
    uv run python scripts/analyze_budget_accuracy.py --model openai/gpt-52--none --comp apex/shortlist_2025 --n-seeds 3
"""

import argparse
import json
import random
from pathlib import Path

import numpy as np


def load_problem_data(output_dir: Path) -> list[dict]:
    """Load all problem JSON files from the output directory."""
    problems = []
    for json_file in sorted(output_dir.glob("*.json"), key=lambda x: int(x.stem)):
        with open(json_file) as f:
            data = json.load(f)
            data["_filename"] = json_file.name
            problems.append(data)
    return problems


def compute_pass_at_k(correct_list: list[bool], k: int, rng: random.Random) -> tuple[bool, list[int]]:
    """
    Compute pass@k by randomly sampling k attempts.
    Returns (passed, sampled_indices).
    """
    n = len(correct_list)
    if k > n:
        return any(correct_list), list(range(n))

    indices = rng.sample(range(n), k)
    passed = any(correct_list[i] for i in indices)
    return passed, indices


def compute_cost_for_indices(detailed_costs: list[dict], indices: list[int]) -> dict:
    """Compute total cost for the given indices."""
    if not detailed_costs:
        return {"cost": 0, "input_tokens": 0, "output_tokens": 0, "reasoning_tokens": 0}

    total_cost = 0
    total_input = 0
    total_output = 0
    total_reasoning = 0

    for i in indices:
        if i < len(detailed_costs):
            dc = detailed_costs[i]
            total_cost += dc.get("cost", 0) or 0
            total_input += dc.get("input_tokens", 0) or 0
            total_output += dc.get("output_tokens", 0) or 0
            total_reasoning += dc.get("reasoning_tokens", 0) or 0

    return {
        "cost": total_cost,
        "input_tokens": total_input,
        "output_tokens": total_output,
        "reasoning_tokens": total_reasoning,
    }


def analyze_model(output_dir: Path, max_power: int = 6, seed: int = 42) -> dict:
    """
    Analyze pass@2^N for N=0 to max_power.
    Returns results dict with metrics for each budget level.
    """
    problems = load_problem_data(output_dir)

    if not problems:
        print(f"No problem files found in {output_dir}")
        return {}

    # Check how many runs we have
    first_problem = problems[0]
    n_runs = first_problem.get("N", len(first_problem.get("correct", [])))
    print(f"Found {len(problems)} problems with up to {n_runs} runs each")

    results = {}

    for power in range(max_power + 1):
        k = 2 ** power
        if k > n_runs:
            print(f"Skipping pass@{k}: only {n_runs} runs available")
            continue

        # Create RNG with base seed for this budget level
        rng = random.Random(seed + power * 1000)

        # For each problem, compute pass@k and costs
        passes = []
        total_cost = 0
        total_input_tokens = 0
        total_output_tokens = 0
        total_reasoning_tokens = 0
        problems_used = 0

        for problem in problems:
            correct = problem.get("correct", [])
            detailed_costs = problem.get("detailed_costs", [])

            if len(correct) < k:
                # Skip problems without enough runs
                continue

            passed, indices = compute_pass_at_k(correct, k, rng)
            passes.append(passed)
            problems_used += 1

            # Compute cost for sampled indices
            cost_info = compute_cost_for_indices(detailed_costs, indices)
            total_cost += cost_info["cost"]
            total_input_tokens += cost_info["input_tokens"]
            total_output_tokens += cost_info["output_tokens"]
            total_reasoning_tokens += cost_info["reasoning_tokens"]

        if passes:
            accuracy = sum(passes) / len(passes)
            results[k] = {
                "budget": k,
                "power": power,
                "n_problems": len(passes),
                "n_correct": sum(passes),
                "accuracy": accuracy,
                "total_cost": total_cost,
                "total_input_tokens": total_input_tokens,
                "total_output_tokens": total_output_tokens,
                "total_reasoning_tokens": total_reasoning_tokens,
                "avg_cost_per_problem": total_cost / len(passes) if passes else 0,
            }

    return results


def analyze_multi_seed(output_dir: Path, max_power: int = 6, n_seeds: int = 3, base_seed: int = 42) -> dict:
    """
    Run analysis with multiple seeds and compute mean ± std.
    """
    problems = load_problem_data(output_dir)
    if not problems:
        print(f"No problem files found in {output_dir}")
        return {}

    first_problem = problems[0]
    n_runs = first_problem.get("N", len(first_problem.get("correct", [])))
    print(f"Found {len(problems)} problems with up to {n_runs} runs each")
    print(f"Running {n_seeds} seeds for variance estimation...")

    # Collect results across seeds
    all_results = []
    for seed_idx in range(n_seeds):
        seed = base_seed + seed_idx * 1000
        results = analyze_model(output_dir, max_power, seed)
        all_results.append(results)

    # Aggregate results
    aggregated = {}
    budgets = sorted(all_results[0].keys())

    for k in budgets:
        accuracies = [r[k]["accuracy"] for r in all_results if k in r]
        costs = [r[k]["total_cost"] for r in all_results if k in r]
        input_tokens = [r[k]["total_input_tokens"] for r in all_results if k in r]
        output_tokens = [r[k]["total_output_tokens"] for r in all_results if k in r]
        reasoning_tokens = [r[k]["total_reasoning_tokens"] for r in all_results if k in r]
        n_correct = [r[k]["n_correct"] for r in all_results if k in r]

        aggregated[k] = {
            "budget": k,
            "n_problems": all_results[0][k]["n_problems"],
            "accuracy_mean": np.mean(accuracies),
            "accuracy_std": np.std(accuracies),
            "n_correct_mean": np.mean(n_correct),
            "n_correct_std": np.std(n_correct),
            "cost_mean": np.mean(costs),
            "cost_std": np.std(costs),
            "input_tokens_mean": np.mean(input_tokens),
            "output_tokens_mean": np.mean(output_tokens),
            "reasoning_tokens_mean": np.mean(reasoning_tokens),
        }

    return aggregated


def print_table(results: dict, model_name: str, multi_seed: bool = False):
    """Print results as a formatted table."""
    print(f"\n{'='*120}")
    print(f"Model: {model_name}")
    print(f"{'='*120}\n")

    if multi_seed:
        # Multi-seed table with mean ± std
        print(f"{'Budget':<10} {'Accuracy':<18} {'Correct':<14} {'Cost ($)':<18} {'Output Tok':<14} {'Reasoning Tok':<14}")
        print("-" * 120)

        for k in sorted(results.keys()):
            r = results[k]
            acc_str = f"{r['accuracy_mean']*100:>5.2f}% ± {r['accuracy_std']*100:>4.2f}%"
            correct_str = f"{r['n_correct_mean']:.1f}/{r['n_problems']}"
            cost_str = f"${r['cost_mean']:>8.4f} ± {r['cost_std']:.2f}"
            print(f"{r['budget']:<10} {acc_str:<18} {correct_str:<14} {cost_str:<18} {r['output_tokens_mean']:>12,.0f}   {r['reasoning_tokens_mean']:>12,.0f}")

        print("-" * 120)

        # Markdown table
        print(f"\n### Markdown Table\n")
        print("| Budget | Accuracy | Correct | Cost ($) | Output Tokens | Reasoning Tokens |")
        print("|-------:|:--------:|:-------:|---------:|--------------:|-----------------:|")
        for k in sorted(results.keys()):
            r = results[k]
            print(f"| {r['budget']} | {r['accuracy_mean']*100:.2f}% ± {r['accuracy_std']*100:.2f}% | {r['n_correct_mean']:.1f}/{r['n_problems']} | ${r['cost_mean']:.4f} | {r['output_tokens_mean']:,.0f} | {r['reasoning_tokens_mean']:,.0f} |")
    else:
        # Single seed table
        print(f"{'Budget':<10} {'Accuracy':<12} {'Correct':<12} {'Cost ($)':<14} {'Output Tok':<14} {'Reasoning Tok':<14}")
        print("-" * 90)

        for k in sorted(results.keys()):
            r = results[k]
            print(f"{r['budget']:<10} {r['accuracy']*100:>6.2f}%     {r['n_correct']:>3}/{r['n_problems']:<6}   ${r['total_cost']:>10.4f}   {r['total_output_tokens']:>12,}   {r['total_reasoning_tokens']:>12,}")

        print("-" * 90)

        # Markdown table
        print(f"\n### Markdown Table\n")
        print("| Budget | Accuracy | Correct | Cost ($) | Output Tokens | Reasoning Tokens |")
        print("|-------:|:--------:|:-------:|---------:|--------------:|-----------------:|")
        for k in sorted(results.keys()):
            r = results[k]
            print(f"| {r['budget']} | {r['accuracy']*100:.2f}% | {r['n_correct']}/{r['n_problems']} | ${r['total_cost']:.4f} | {r['total_output_tokens']:,} | {r['total_reasoning_tokens']:,} |")


def main():
    parser = argparse.ArgumentParser(description="Analyze pass@2^N metrics and costs")
    parser.add_argument("--model", type=str, default="openai/gpt-52--none",
                        help="Model identifier (e.g., openai/gpt-52--none)")
    parser.add_argument("--comp", type=str, default="apex/shortlist_2025",
                        help="Competition path (e.g., apex/shortlist_2025)")
    parser.add_argument("--max-power", type=int, default=6,
                        help="Maximum power of 2 (default: 6 for pass@64)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for sampling")
    parser.add_argument("--n-seeds", type=int, default=1,
                        help="Number of seeds to run for variance estimation (default: 1)")
    parser.add_argument("--output-json", type=str, default=None,
                        help="Optional: save results to JSON file")

    args = parser.parse_args()

    output_dir = Path("outputs") / args.comp / args.model

    if not output_dir.exists():
        print(f"Error: Output directory not found: {output_dir}")
        return

    if args.n_seeds > 1:
        results = analyze_multi_seed(output_dir, args.max_power, args.n_seeds, args.seed)
        multi_seed = True
    else:
        results = analyze_model(output_dir, args.max_power, args.seed)
        multi_seed = False

    if results:
        print_table(results, args.model, multi_seed=multi_seed)

        if args.output_json:
            # Convert numpy types to Python types for JSON serialization
            json_results = {}
            for k, v in results.items():
                json_results[k] = {key: float(val) if isinstance(val, (np.floating, np.integer)) else val
                                   for key, val in v.items()}
            with open(args.output_json, "w") as f:
                json.dump(json_results, f, indent=2)
            print(f"\nResults saved to {args.output_json}")


if __name__ == "__main__":
    main()
