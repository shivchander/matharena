#!/usr/bin/env python3
"""
Simplified pipeline analysis: accuracy vs token budget at powers of 2.

Usage:
    uv run python scripts/analyze_pipeline_budget.py \
        --comp apex/shortlist_2025 \
        --plan-gen openai/gpt-52--batch-plan-gen \
        --plan-score openai/gpt-52--batch-plan-scoring \
        --cond-resp openai/gpt-52--batch-cond-resp
"""

import argparse
import json
import random
import statistics
from pathlib import Path


def load_outputs(comp: str, model: str) -> dict:
    """Load all output files for a model."""
    output_dir = Path("outputs") / comp / model
    outputs = {}
    if not output_dir.exists():
        return outputs
    for f in output_dir.glob("*.json"):
        problem_idx = int(f.stem)
        with open(f) as fp:
            outputs[problem_idx] = json.load(fp)
    return outputs


def extract_tokens(output: dict) -> int:
    """Extract total output tokens from an output dict."""
    cost = output.get("cost", {})
    return (cost.get("output_tokens", 0) or 0) + (cost.get("reasoning_tokens", 0) or 0)


def analyze_pipeline(comp: str, plan_gen: str, plan_score: str, cond_resp: str, n_seeds: int = 5, base_seed: int = 42) -> dict:
    """Analyze pipeline: accuracy and tokens at each budget level with error bars."""

    # Load outputs
    pg_outputs = load_outputs(comp, plan_gen)
    ps_outputs = load_outputs(comp, plan_score)
    cr_outputs = load_outputs(comp, cond_resp)

    # Get common problems
    problems = sorted(set(pg_outputs.keys()) & set(ps_outputs.keys()) & set(cr_outputs.keys()))

    if not problems:
        print("No common problems found!")
        return {}

    # Compute upstream costs (plan gen + scoring) - fixed for all budgets
    upstream_tokens = 0
    for idx in problems:
        upstream_tokens += extract_tokens(pg_outputs[idx])
        upstream_tokens += extract_tokens(ps_outputs[idx])

    # Get number of runs per problem
    first_cr = cr_outputs[problems[0]]
    n_runs = first_cr.get("N", len(first_cr.get("correct", [])))

    print(f"Problems: {len(problems)}")
    print(f"Runs per problem: {n_runs}")
    print(f"Upstream tokens (plan gen + scoring): {upstream_tokens:,}")
    print(f"Seeds for variance: {n_seeds}")
    print()

    results = {}

    # Analyze each budget level (powers of 2)
    for power in range(7):  # 1, 2, 4, 8, 16, 32, 64
        k = 2 ** power
        if k > n_runs:
            break

        # Run multiple seeds for variance estimation
        seed_accuracies = []
        seed_response_tokens = []

        for seed_idx in range(n_seeds):
            seed = base_seed + seed_idx * 1000 + power * 100
            rng = random.Random(seed)

            passes = []
            response_tokens = 0

            for idx in problems:
                cr = cr_outputs[idx]
                correct = cr.get("correct", [])
                detailed_costs = cr.get("detailed_costs", [])

                if len(correct) < k:
                    continue

                # Sample k responses
                indices = rng.sample(range(len(correct)), k)
                passed = any(correct[i] for i in indices)
                passes.append(passed)

                # Sum tokens for sampled responses
                for i in indices:
                    if i < len(detailed_costs):
                        dc = detailed_costs[i]
                        response_tokens += (dc.get("output_tokens", 0) or 0)
                        response_tokens += (dc.get("reasoning_tokens", 0) or 0)

            if passes:
                seed_accuracies.append(sum(passes) / len(passes))
                seed_response_tokens.append(response_tokens)

        if seed_accuracies:
            acc_mean = statistics.mean(seed_accuracies)
            acc_std = statistics.stdev(seed_accuracies) if len(seed_accuracies) > 1 else 0
            resp_tokens_mean = statistics.mean(seed_response_tokens)
            total_tokens = upstream_tokens + resp_tokens_mean

            results[k] = {
                "budget": k,
                "accuracy_mean": acc_mean,
                "accuracy_std": acc_std,
                "n_correct_mean": acc_mean * len(problems),
                "n_problems": len(problems),
                "total_tokens": total_tokens,
                "upstream_tokens": upstream_tokens,
                "response_tokens": resp_tokens_mean,
            }

    return results


def print_results(results: dict, models: dict):
    """Print results as a simple table."""
    print("=" * 70)
    print(f"Plan Gen:   {models['plan_gen']}")
    print(f"Plan Score: {models['plan_score']}")
    print(f"Cond Resp:  {models['cond_resp']}")
    print("=" * 70)
    print()

    print(f"{'Budget':<10} {'Accuracy':<20} {'Correct':<12} {'Response Tok':<18} {'Total Tokens':<18}")
    print("-" * 90)

    for k in sorted(results.keys()):
        r = results[k]
        acc_str = f"{r['accuracy_mean']*100:.2f}% ± {r['accuracy_std']*100:.2f}%"
        correct_str = f"{r['n_correct_mean']:.1f}/{r['n_problems']}"
        print(f"{r['budget']:<10} {acc_str:<20} {correct_str:<12} {r['response_tokens']:>15,.0f}   {r['total_tokens']:>15,.0f}")

    print("-" * 90)
    print()

    # Markdown table
    print("### Markdown Table")
    print()
    print("| Budget | Accuracy | Correct | Response Tokens | Total Tokens |")
    print("|-------:|---------:|--------:|----------------:|-------------:|")
    for k in sorted(results.keys()):
        r = results[k]
        print(f"| {r['budget']} | {r['accuracy_mean']*100:.2f}% ± {r['accuracy_std']*100:.2f}% | {r['n_correct_mean']:.1f}/{r['n_problems']} | {r['response_tokens']:,.0f} | {r['total_tokens']:,.0f} |")


def main():
    parser = argparse.ArgumentParser(description="Analyze pipeline: accuracy vs token budget")
    parser.add_argument("--comp", required=True, help="Competition path")
    parser.add_argument("--plan-gen", required=True, help="Plan generation model")
    parser.add_argument("--plan-score", required=True, help="Plan scoring model")
    parser.add_argument("--cond-resp", required=True, help="Conditioned response model")
    parser.add_argument("--n-seeds", type=int, default=5, help="Number of seeds for variance estimation")
    parser.add_argument("--seed", type=int, default=42, help="Base random seed")

    args = parser.parse_args()

    results = analyze_pipeline(
        args.comp,
        args.plan_gen,
        args.plan_score,
        args.cond_resp,
        n_seeds=args.n_seeds,
        base_seed=args.seed,
    )

    if results:
        print_results(results, {
            "plan_gen": args.plan_gen,
            "plan_score": args.plan_score,
            "cond_resp": args.cond_resp,
        })


if __name__ == "__main__":
    main()
