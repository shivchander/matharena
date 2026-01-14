# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

MathArena is an LLM evaluation platform for assessing language models on recent mathematics competitions and olympiads (NeurIPS D&B '25). It evaluates models against uncontaminated math competitions including AIME, IMO, USAMO, Project Euler, and more.

- Platform: https://matharena.ai/
- HuggingFace datasets and evaluation logs: https://huggingface.co/MathArena

## Commands

### Installation
```bash
# Using UV (recommended)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Alternative: Conda
conda create -n matharena python=3.12 && conda activate matharena && pip install -e .
```

### Running Evaluations
```bash
# Basic run
uv run python scripts/run.py --comp aime/aime_2025 --models openai/gpt-4o

# With parameters: --n (runs per problem), --problems (specific problems), --redo-all (ignore existing)
uv run python scripts/run.py --comp path/to/comp --models model1 model2 --n 4 --problems 1 2 3 --redo-all

# Re-grade existing runs (updates parsing/grading without re-running models)
uv run python scripts/regrade.py --comps euler/euler
```

### Testing
```bash
uv run pytest tests/test_code_execution.py
```

### Web Dashboard (inspect runs)
```bash
uv run python scripts/app.py --comp path/to/competition --port 5001
# Access at http://localhost:5001/
```

### Curation Scripts
```bash
# Verify LaTeX compilation
uv run python scripts/curation/check_latex.py --comp path/to/competition

# Test parser changes against all existing outputs
uv run python scripts/curation/test_parser_changes.py

# Verify parser with judge model
uv run python scripts/curation/judge_parser.py --comp aime/aime_2025

# Upload to HuggingFace
uv run python scripts/curation/upload_competition.py --org your_org --repo-name your_repo_name --comp path/to/competition
```

### Analysis Scripts
```bash
# Leaderboard with confidence intervals
uv run python scripts/extraction/leaderboard.py --comps path/to/comp1 path/to/comp2 --compute-variance

# Compare competitions
uv run python scripts/extraction/comparison.py --old-comps aime/aime_2024 --new-comps aime/aime_2025
```

### Docker (for code execution sandbox)
```bash
docker build -t matharena-docker docker/
```

## Architecture

### Core Pipeline Flow
1. **Runner** (`src/matharena/runner.py`): Orchestrates evaluations - loads competition/problems, instantiates solver, executes batched queries
2. **Solver** (`src/matharena/solvers/`): Either `PureModelSolver` (single-turn) or `AgentPool` (multi-turn with tools)
3. **API Client** (`src/matharena/api_client.py`): Unified interface for 10+ LLM providers with batching, retry, and cost tracking
4. **Parser** (`src/matharena/parser.py`): Extracts answers from model output (finds `\boxed{}` content, converts LaTeX to SymPy)
5. **Grader** (`src/matharena/grader.py`): Compares parsed answers against gold (exact match or symbolic equivalence)
6. **Runs** (`src/matharena/runs.py`): Serializes results to standardized JSON format in `outputs/`

### Key Directories
- `configs/competitions/`: Competition YAML configs (dataset path, instructions, parsing mode)
- `configs/models/`: Model YAML configs organized by provider (API type, costs, tokens)
- `configs/agent_scaffolds/`: Agent workflow configs (selfcheck, deepseek_math)
- `outputs/`: Saved evaluation runs (JSON)
- `logs/status/`: Run progress tracking
- `logs/requests/`: Verbatim API request logs
- `data/`: Local competition data (problems/*.tex, answers.csv)

### Adding New Models
Create YAML in `configs/models/{provider}/{model}.yaml`:
```yaml
model: "model-name"  # Append --[low/medium/high] for OpenAI reasoning effort
api: "openai|anthropic|google|deepseek|xai|together|glm|vllm"
human_readable_id: "unique-id"
max_tokens: 16000
read_cost: 2.5  # per million tokens
write_cost: 10
```

### Adding New Competitions
1. Create YAML in `configs/competitions/{name}.yaml`:
```yaml
instruction: "Solve... put final answer in \\boxed{}"
strict_parsing: true
n_problems: 30
date: "2025-01-13"
dataset_path: "hf://organization/dataset"
```
2. Dataset must have columns: `problem_idx`, `problem`, `answer` (optional: `points`, `grading_scheme`)

### Parser Manual Overrides
For edge cases where automatic parsing fails, add mappings in `src/matharena/parse_manual.py`.

## Environment Variables

Required API keys (set based on which providers you use):
- `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GOOGLE_API_KEY`
- `DEEPSEEK_API_KEY`, `XAI_API_KEY`, `TOGETHER_API_KEY`
- `OPENROUTER_API_KEY`, `GLM_API_KEY`

## Code Style

- Black formatter with 120 character line length
- Flake8 ignores: E203, E266, E501, W503

## Warning Indicators in Web Dashboard

When viewing runs at localhost:5001:
- 💀: Parser threw an error
- ⚠️: Correct answer may be present but wasn't extracted
- ❕: Model likely hit max token limit
