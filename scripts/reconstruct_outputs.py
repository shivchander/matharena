#!/usr/bin/env python3
"""
Reconstruct output JSONs from checkpoint files.

Usage:
    python scripts/reconstruct_outputs.py \
        --checkpoint-dir "agent_checkpoints copy" \
        --checkpoint-prefix "plan_cond_resp_gpt-5.2--none" \
        --comp apex/shortlist_2025 \
        --output-model openai/gpt-52--cond-resp \
        --n 64
"""

import argparse
import json
import os
import re
from collections import defaultdict
from pathlib import Path

from datasets import load_dataset
from loguru import logger

from matharena.parser import extract_boxed_answer_parse
from matharena.grader import check_answers
from matharena.parser import parse_answer


def load_competition_problems(comp_path: str) -> dict:
    """Load problems from the competition config."""
    config_path = Path("configs/competitions") / f"{comp_path}.yaml"

    import yaml
    with open(config_path) as f:
        config = yaml.safe_load(f)

    dataset_path = config.get("dataset_path", "")
    if dataset_path.startswith("hf://"):
        dataset_path = dataset_path[5:]

    # Load dataset (try test first, fallback to train)
    try:
        dataset = load_dataset(dataset_path, split="test")
    except ValueError:
        dataset = load_dataset(dataset_path, split="train")

    problems = {}
    for row in dataset:
        idx = row["problem_idx"]
        problems[idx] = {
            "idx": idx,
            "problem": row["problem"],
            "gold_answer": row["answer"],
            "source": row.get("source", ""),
            "types": row.get("types", []),
        }

    return problems


def load_checkpoints(checkpoint_dir: str, prefix: str) -> dict:
    """Load all checkpoints matching the prefix, organized by problem_idx."""
    checkpoints = defaultdict(list)

    pattern = re.compile(rf"{re.escape(prefix)}_(\d+)_([a-f0-9]+)_r(\d+)\.json")

    for filename in os.listdir(checkpoint_dir):
        match = pattern.match(filename)
        if match:
            problem_idx = int(match.group(1))
            run_idx = int(match.group(3))

            filepath = os.path.join(checkpoint_dir, filename)
            with open(filepath) as f:
                checkpoint = json.load(f)

            checkpoints[problem_idx].append({
                "run_idx": run_idx,
                "checkpoint": checkpoint,
                "filename": filename,
            })

    # Sort by run_idx
    for problem_idx in checkpoints:
        checkpoints[problem_idx].sort(key=lambda x: x["run_idx"])

    return checkpoints


def extract_conversation_from_checkpoint(checkpoint: dict) -> list:
    """Extract the conversation (messages) from a checkpoint."""
    history = checkpoint.get("history", [])

    # Try different step names in order of preference
    step_names = ["solution_generation", "plan_generated", "generation_summary"]

    for step_name in step_names:
        for step in history:
            if step.get("step") == step_name:
                messages = step.get("messages", [])
                if messages:
                    return messages

    # Fallback: return last step's messages
    if history:
        return history[-1].get("messages", [])

    return []


def get_assistant_response(messages: list) -> str:
    """Get the last assistant response from a conversation."""
    for msg in reversed(messages):
        if msg.get("role") == "assistant":
            return msg.get("content", "")
    return ""


def grade_answer(response: str, gold_answer: str) -> tuple:
    """Extract and grade an answer from a response."""
    try:
        result = extract_boxed_answer_parse(response)
        if result is not None:
            parsed, warning = result
            gold_parsed, _ = parse_answer(gold_answer)
            is_correct = check_answers(parsed, gold_parsed)
            # Convert warning to string if it's not None
            warning_str = str(warning) if warning is not None else None
            return str(parsed), is_correct, warning_str
        return None, False, "no_boxed_answer"
    except Exception as e:
        return None, False, f"parse_error: {e}"


def reconstruct_output(problem: dict, checkpoints: list, n: int) -> dict:
    """Reconstruct an output JSON from checkpoints."""
    messages = []
    answers = []
    correct = []
    warnings = []
    detailed_costs = []
    history = []

    total_cost = {"cost": 0, "input_tokens": 0, "output_tokens": 0, "time": 0}

    # Use only the first n checkpoints
    checkpoints = checkpoints[:n]

    for cp_data in checkpoints:
        checkpoint = cp_data["checkpoint"]

        # Extract conversation
        convo = extract_conversation_from_checkpoint(checkpoint)
        messages.append(convo)

        # Get response and grade
        response = get_assistant_response(convo)
        answer, is_correct, warning = grade_answer(response, problem["gold_answer"])

        answers.append(answer)
        correct.append(is_correct)
        warnings.append(warning)

        # Extract costs
        dc = checkpoint.get("detailed_cost", {})
        detailed_costs.append(dc)

        for key in ["cost", "input_tokens", "output_tokens", "time"]:
            total_cost[key] += dc.get(key, 0) or 0

        # Extract history
        history.append(checkpoint.get("history", []))

    # Build output
    output = {
        "idx": problem["idx"],
        "problem": problem["problem"],
        "gold_answer": problem["gold_answer"],
        "source": problem.get("source", ""),
        "types": problem.get("types", []),
        "N": len(checkpoints),
        "cost": total_cost,
        "pass_at_1": correct[0] if correct else False,
        "answers": answers,
        "correct": correct,
        "warnings": warnings,
        "messages": messages,
        "judgment": None,
        "history": history,
        "detailed_costs": detailed_costs,
    }

    return output


def main():
    parser = argparse.ArgumentParser(description="Reconstruct outputs from checkpoints")
    parser.add_argument("--checkpoint-dir", required=True, help="Directory with checkpoint files")
    parser.add_argument("--checkpoint-prefix", required=True, help="Prefix for checkpoint files")
    parser.add_argument("--comp", required=True, help="Competition path")
    parser.add_argument("--output-model", required=True, help="Output model path (e.g., openai/gpt-52--cond-resp)")
    parser.add_argument("--n", type=int, default=64, help="Number of runs per problem")

    args = parser.parse_args()

    # Load problems
    logger.info(f"Loading problems from {args.comp}")
    problems = load_competition_problems(args.comp)
    logger.info(f"Loaded {len(problems)} problems")

    # Load checkpoints
    logger.info(f"Loading checkpoints from {args.checkpoint_dir}")
    checkpoints = load_checkpoints(args.checkpoint_dir, args.checkpoint_prefix)
    logger.info(f"Found checkpoints for {len(checkpoints)} problems")

    # Create output directory
    output_dir = Path("outputs") / args.comp / args.output_model
    output_dir.mkdir(parents=True, exist_ok=True)

    # Reconstruct outputs
    for problem_idx, problem in problems.items():
        if problem_idx not in checkpoints:
            logger.warning(f"No checkpoints for problem {problem_idx}")
            continue

        cp_list = checkpoints[problem_idx]
        if len(cp_list) < args.n:
            logger.warning(f"Problem {problem_idx}: only {len(cp_list)}/{args.n} checkpoints")

        output = reconstruct_output(problem, cp_list, args.n)

        output_path = output_dir / f"{problem_idx}.json"
        with open(output_path, "w") as f:
            json.dump(output, f, indent=2)

        n_correct = sum(output["correct"])
        logger.info(f"Problem {problem_idx}: {n_correct}/{output['N']} correct, saved to {output_path}")

    logger.info("Done!")


if __name__ == "__main__":
    main()
