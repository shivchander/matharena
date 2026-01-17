#!/usr/bin/env python
"""Analyze results from model runs - compute accuracy and cost metrics."""

import argparse
import json
import os
from pathlib import Path


def analyze_results(output_dir: str, verbose: bool = True):
    """
    Analyze all result JSON files in a directory.

    Args:
        output_dir: Path to directory containing result JSON files
        verbose: Whether to print per-problem results

    Returns:
        dict with accuracy, cost, and token metrics
    """
    total_cost = 0
    total_input_tokens = 0
    total_output_tokens = 0
    total_reasoning_tokens = 0
    correct = 0
    total = 0
    total_time = 0

    results = []

    # Sort by problem index
    files = [f for f in os.listdir(output_dir) if f.endswith('.json')]
    files = sorted(files, key=lambda x: int(x.replace('.json', '')))

    for fname in files:
        with open(os.path.join(output_dir, fname)) as f:
            data = json.load(f)

        prob_idx = data['idx']
        gold = data['gold_answer']
        answers = data.get('answers', [])
        correct_list = data.get('correct', [])
        is_correct = correct_list[0] if correct_list else False
        cost = data.get('cost', {})

        total_cost += cost.get('cost', 0)
        total_input_tokens += cost.get('input_tokens', 0)
        total_output_tokens += cost.get('output_tokens', 0)
        total_reasoning_tokens += cost.get('reasoning_tokens', 0) or 0
        total_time += cost.get('time', 0)

        if is_correct:
            correct += 1
        total += 1

        status = '✓' if is_correct else '✗'
        answer = answers[0] if answers else 'N/A'
        results.append((prob_idx, answer, gold, status, is_correct))

    # Compute metrics
    accuracy = correct / total if total > 0 else 0
    total_tokens = total_input_tokens + total_output_tokens

    metrics = {
        'problems': total,
        'correct': correct,
        'accuracy': accuracy,
        'total_cost': total_cost,
        'input_tokens': total_input_tokens,
        'output_tokens': total_output_tokens,
        'reasoning_tokens': total_reasoning_tokens,
        'total_tokens': total_tokens,
        'total_time': total_time,
        'results': results,
    }

    return metrics


def print_report(metrics: dict, output_dir: str, verbose: bool = True):
    """Print a formatted report of the metrics."""
    model_name = '/'.join(Path(output_dir).parts[-2:])

    print(f'=== RESULTS: {model_name} ===')
    print()
    print(f"Problems: {metrics['problems']}")
    print(f"Correct: {metrics['correct']}/{metrics['problems']} = {100*metrics['accuracy']:.1f}%")
    print()
    print(f"Total cost: ${metrics['total_cost']:.2f}")
    print(f"Input tokens: {metrics['input_tokens']:,}")
    print(f"Output tokens: {metrics['output_tokens']:,}")
    if metrics['reasoning_tokens'] > 0:
        print(f"Reasoning tokens: {metrics['reasoning_tokens']:,}")
    print(f"Total tokens: {metrics['total_tokens']:,}")
    print(f"Total time: {metrics['total_time']:.1f}s ({metrics['total_time']/60:.1f}min)")

    if verbose:
        print()
        print('=== PER-PROBLEM RESULTS ===')
        for prob_idx, answer, gold, status, _ in metrics['results']:
            # Truncate long answers
            answer_str = str(answer)[:20] + '...' if len(str(answer)) > 20 else str(answer)
            gold_str = str(gold)[:20] + '...' if len(str(gold)) > 20 else str(gold)
            print(f'P{prob_idx:2d}: {status} (got {answer_str}, gold {gold_str})')


def main():
    parser = argparse.ArgumentParser(description='Analyze model run results')
    parser.add_argument('output_dir', type=str,
                        help='Path to output directory containing JSON results')
    parser.add_argument('--quiet', '-q', action='store_true',
                        help='Only show summary, not per-problem results')
    parser.add_argument('--json', action='store_true',
                        help='Output as JSON instead of formatted text')

    args = parser.parse_args()

    if not os.path.isdir(args.output_dir):
        print(f"Error: {args.output_dir} is not a directory")
        return 1

    metrics = analyze_results(args.output_dir, verbose=not args.quiet)

    if args.json:
        # Remove non-serializable results for JSON output
        output = {k: v for k, v in metrics.items() if k != 'results'}
        output['per_problem'] = [
            {'idx': r[0], 'answer': r[1], 'gold': r[2], 'correct': r[4]}
            for r in metrics['results']
        ]
        print(json.dumps(output, indent=2))
    else:
        print_report(metrics, args.output_dir, verbose=not args.quiet)

    return 0


if __name__ == '__main__':
    exit(main())
