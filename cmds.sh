#!/bin/bash

# Source environment variables
set -a && source .env && set +a

# ============================================================
# PLAN TOURNAMENT PIPELINE (apex/shortlist_2025)
# ============================================================

# STEP 1: Generate 64 plans + run tournament
# API calls per problem: 64 (plans) + 63 (matches) = 127
# Estimated cost: ~$2-3 per problem
set -a && source .env && set +a && uv run python scripts/run.py \
    --comp apex/shortlist_2025 \
    --models openai/gpt-52--none-plan-tournament-k64 \
    --n 1

# STEP 2: Generate 4 solutions conditioned on winning plans + tournament
# API calls per problem: 4 (solutions) + 3 (matches) = 7
# Run after Step 1 completes
# uv run python scripts/run.py \
#     --comp apex/shortlist_2025 \
#     --models openai/gpt-52--none-conditioned-solver-m4-k64 \
#     --n 1

# ============================================================
# SMALLER TESTS (K=4, M=4)
# ============================================================

# Plan tournament with K=4 (quick test)
# uv run python scripts/run.py \
#     --comp apex/shortlist_2025 \
#     --models openai/gpt-52--none-plan-tournament-k4 \
#     --problems 1 \
#     --n 1

# Conditioned solver with M=4 (requires K=4 plans first)
# uv run python scripts/run.py \
#     --comp apex/shortlist_2025 \
#     --models openai/gpt-52--none-conditioned-solver-m4 \
#     --problems 1 \
#     --n 1

# ============================================================
# LEGACY COMMANDS (Best-of-N on apex_2025)
# ============================================================

# GPT-5.2 with no reasoning
# uv run python scripts/run.py --comp apex/apex_2025 --models openai/gpt-52--none --n 1

# Bo8
# uv run python scripts/run.py --comp apex/apex_2025 --models openai/gpt-52--none-best-of-8 --n 1

# Bo32
# uv run python scripts/run.py --comp apex/apex_2025 --models openai/gpt-52--none-best-of-32 --n 1

# Bo64
# uv run python scripts/run.py --comp apex/apex_2025 --models openai/gpt-52--none-best-of-64 --n 1


uv run python scripts/analyze_planning_pipeline.py \
      --comp apex/shortlist_2025 \
      --plan-gen anthropic/claude-sonnet-45-or--plan-gen \
      --plan-score anthropic/claude-sonnet-45-or--plan-scoring \
      --cond-resp anthropic/claude-sonnet-45-or--cond-resp \
      --sample-k 8