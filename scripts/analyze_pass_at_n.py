"""
Analyze pass@N for Best-of-N runs.

This script checks if any of the N generated solutions is correct,
computing an upper-bound pass@N metric. Useful for diagnosing whether
the critic/judge is failing to select correct solutions.

Usage:
    uv run python scripts/analyze_pass_at_n.py --output-dir outputs/apex/apex_2025/openai/gpt-52--none-best-of-64
"""

import argparse
import json
import os
from collections import defaultdict

from matharena.grader import check_answers, extract_and_grade, extract_answer
from matharena.parser import extract_boxed_answer_parse, parse_answer


def analyze_solutions(output_dir: str, verbose: bool = True):
    """Analyze all solutions in output files to compute pass@N."""

    # Get all problem files
    files = sorted(
        [f for f in os.listdir(output_dir) if f.endswith('.json')],
        key=lambda x: int(x.replace('.json', ''))
    )

    if not files:
        print(f"No JSON files found in {output_dir}")
        return

    print("=" * 80)
    print(f"PASS@N ANALYSIS - {output_dir}")
    print("=" * 80)

    results = []

    for fname in files:
        fpath = os.path.join(output_dir, fname)
        with open(fpath) as f:
            data = json.load(f)

        problem_idx = data['idx']
        gold_answer = data['gold_answer']
        selected_answer = data['answers'][0] if data['answers'] else None
        selected_correct = data['correct'][0] if data['correct'] else False

        # Get all solutions from history
        history = data.get('history', [[]])[0]
        gen_summary = None
        judge_summary = None
        selection = None

        for h in history:
            if h.get('step') == 'generation_summary':
                gen_summary = h
            elif h.get('step') == 'judging_summary':
                judge_summary = h
            elif h.get('step') == 'selection':
                selection = h

        if not gen_summary:
            print(f"Problem {problem_idx}: No generation_summary found!")
            continue

        solutions = gen_summary.get('solutions', [])
        n_solutions = len(solutions)
        scores = judge_summary.get('scores', []) if judge_summary else []
        best_idx = selection.get('best_index', -1) if selection else -1

        # Parse the gold answer first
        try:
            gold_parsed, _ = parse_answer(gold_answer)
        except Exception:
            gold_parsed = gold_answer  # Fall back to string

        # Parse and grade each solution
        correct_indices = []
        answers_found = []

        for i, sol in enumerate(solutions):
            try:
                # Extract boxed answer (returns tuple of (answer, warning))
                result = extract_boxed_answer_parse(sol)
                if result is not None:
                    parsed, warning = result
                    # Check if correct using parsed gold
                    is_correct = check_answers(parsed, gold_parsed)
                    answers_found.append((i, str(parsed), is_correct))
                    if is_correct:
                        correct_indices.append(i)
                else:
                    answers_found.append((i, "NO_BOXED", False))
            except Exception as e:
                answers_found.append((i, f"PARSE_ERROR", False))

        pass_n = len(correct_indices) > 0

        result = {
            'problem_idx': problem_idx,
            'gold_answer': gold_answer,
            'n_solutions': n_solutions,
            'n_correct': len(correct_indices),
            'pass_n': pass_n,
            'selected_idx': best_idx,
            'selected_answer': selected_answer,
            'selected_correct': selected_correct,
            'correct_indices': correct_indices,
            'scores': scores,
        }
        results.append(result)

        if verbose:
            print(f"\n{'='*60}")
            print(f"Problem {problem_idx}: Gold = {gold_answer}")
            print(f"  Selected idx={best_idx}, answer={selected_answer}, correct={selected_correct}")
            print(f"  Pass@{n_solutions}: {pass_n} ({len(correct_indices)}/{n_solutions} correct)")

            if correct_indices:
                print(f"  CORRECT solutions at indices: {correct_indices}")
                for idx in correct_indices[:3]:
                    score = scores[idx] if idx < len(scores) else "?"
                    ans = answers_found[idx][1][:50] if idx < len(answers_found) else "?"
                    print(f"    - idx {idx}: score={score}, answer={ans}")

                if best_idx >= 0 and best_idx < len(scores):
                    print(f"  Selected solution score: {scores[best_idx]}")
                    correct_scores = [scores[i] for i in correct_indices if i < len(scores)]
                    if correct_scores:
                        print(f"  Correct solution scores: {correct_scores[:5]}")
                        max_correct_score = max(correct_scores)
                        if scores[best_idx] < max_correct_score:
                            print(f"  WARNING: Critic missed higher-scoring correct solution!")
            else:
                # Show sample answers
                unique_answers = defaultdict(list)
                for i, ans, _ in answers_found:
                    unique_answers[ans].append(i)
                print(f"  Sample answers from {n_solutions} solutions:")
                for ans, indices in list(unique_answers.items())[:5]:
                    print(f"    - '{ans[:40]}' (found {len(indices)}x)")

    # Summary
    total_problems = len(results)
    total_pass_n = sum(1 for r in results if r['pass_n'])
    total_selected_correct = sum(1 for r in results if r['selected_correct'])

    n_solutions = results[0]['n_solutions'] if results else 0

    print(f"\n{'='*80}")
    print(f"SUMMARY")
    print(f"{'='*80}")
    print(f"  Problems analyzed: {total_problems}")
    print(f"  Pass@{n_solutions} (upper bound): {total_pass_n}/{total_problems} = {total_pass_n/total_problems*100:.1f}%")
    print(f"  Pass@1 (selected):  {total_selected_correct}/{total_problems} = {total_selected_correct/total_problems*100:.1f}%")

    if total_pass_n > total_selected_correct:
        missed = total_pass_n - total_selected_correct
        print(f"\n  CRITIC FAILURE: {missed} problems had correct solutions but wrong one was selected!")
        print(f"  Problems with missed correct solutions:")
        for r in results:
            if r['pass_n'] and not r['selected_correct']:
                correct_scores = [r['scores'][i] for i in r['correct_indices'] if i < len(r['scores'])]
                selected_score = r['scores'][r['selected_idx']] if r['selected_idx'] < len(r['scores']) else "?"
                print(f"    - Problem {r['problem_idx']}: selected score={selected_score}, correct scores={correct_scores[:3]}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Analyze pass@N for Best-of-N runs")
    parser.add_argument("--output-dir", required=True, help="Directory containing output JSON files")
    parser.add_argument("--quiet", action="store_true", help="Only show summary")
    args = parser.parse_args()

    analyze_solutions(args.output_dir, verbose=not args.quiet)


if __name__ == "__main__":
    main()
