#!/bin/bash
# Modular Agent Pipeline Commands
# Load environment variables first: set -a && source .env && set +a

# =============================================================================
# PATH A: Direct Best-of-N (No Planning)
# =============================================================================

# Step 1: Generate N responses with PureModelSolver
set -a && source .env && set +a && uv run python scripts/run.py --comp apex/shortlist_2025 --models openai/gpt-52--none --n 64

# Step 2: Score and select the best response
set -a && source .env && set +a && uv run python scripts/run.py --comp apex/shortlist_2025 --models openai/gpt-52--resp-scoring-direct --n 1

# =============================================================================
# PATH B: Plan-Conditioned Pipeline
# =============================================================================

# Step 1: Generate K plans
set -a && source .env && set +a && uv run python scripts/run.py --comp apex/shortlist_2025 --models openai/gpt-52--plan-gen --n 64

# Step 2: Score plans and select the best
set -a && source .env && set +a && uv run python scripts/run.py --comp apex/shortlist_2025 --models openai/gpt-52--plan-scoring --n 1

# Step 3: Generate M solutions conditioned on the best plan
set -a && source .env && set +a && uv run python scripts/run.py --comp apex/shortlist_2025 --models openai/gpt-52--cond-resp --n 4

# Step 4: Score solutions and select the best
set -a && source .env && set +a && uv run python scripts/run.py --comp apex/shortlist_2025 --models openai/gpt-52--resp-scoring --n 1



uv run python scripts/analyze_budget_accuracy.py \
      --model anthropic/claude-sonnet-45-or-thinking \
      --comp apex/shortlist_2025 \
      --n-seeds 3 
      --output-json results.json