# Plan Generation Architecture for Test-Time Compute Scaling - Akash (read the last section first)

## Motivation

Reasoning models are expensive, and their reasoning traces often waste tokens on inefficient exploration. This document outlines strategies for using inference-time scaling (ITS) to generate efficient, high-quality reasoning plans before full problem solving.

**Core Hypothesis**: By separating plan generation from solution execution, we can:
1. Generate diverse, high-quality reasoning strategies more efficiently
2. Avoid redundant token usage in solution generation
3. Scale test-time compute more controllably

## Existing Architecture

### Current Plan-Then-Solve Pipeline

The branch already implements a modular pipeline:

**Stage 1: Plan Generation**
- Generate K plans in parallel
- Options:
  - Tournament selection (`PlanTournamentAgent`)
  - Scoring-based selection (`PlanScoringAgent`)

**Stage 2: Conditioned Solution Generation**
- Load winning plan from Stage 1
- Generate M solutions conditioned on that plan
- Select best solution via tournament or scoring

**Configs**: See `configs/agent_scaffolds/plan_*.yaml` and model configs like `gpt-52--plan-gen.yaml`

**Limitations**:
- Plans generated independently (no iterative refinement)
- No natural branching at decision points
- Fixed diversity via parallel sampling only

## Proposed: Iterative Plan Generation Strategies

We propose two complementary approaches for generating refined, diverse plans through structured iteration.

---

## Strategy 1: Sequential Refinement Agent

**Concept**: Start with an initial plan, then iteratively critique and refine it through up to `max_refinements` rounds, with early stopping when the plan is deemed sufficient.

### Architecture

```
Initial Plan Generation
    ↓
┌─→ Plan Assessment ─→ [Stop Decision]
│       ↓ (continue)
│   Refinement
│       ↓
└─ Updated Plan ─┐
                 ↓
            [reaches max_refinements or stops]
                 ↓
            Final Plan
```

### Workflow

1. **Initial Generation**: Generate a candidate plan for the problem
2. **Assessment Loop** (up to `max_refinements` iterations):
   - **Critique**: Model reviews current plan and identifies weaknesses, gaps, or alternative approaches
   - **Decision**: Model decides whether to stop (plan is sufficient) or continue refinement
   - **Refinement**: If continuing, model generates an improved version of the plan
3. **Output**: Final refined plan

### Design Decisions

- **Decision mechanism**: Embedded in critique turn - model outputs both critique and stop/continue decision in structured format
- **Prompt structure**:
  ```
  Current plan: {plan}

  Critique this plan. Identify any gaps, errors, or areas for improvement.
  Then decide: should we refine further or is this plan sufficient?

  Output format:
  CRITIQUE: [your analysis]
  DECISION: [STOP/CONTINUE]
  REFINEMENT: [if CONTINUE, provide improved plan]
  ```

### Configuration Parameters

- `max_refinements`: Maximum number of critique-refine iterations (e.g., 4, 8, 16)
- `generation_pool_size`: Parallel workers for initial generation
- Model and temperature settings

### Example Config: `sequential_plan_refinement.yaml`

```yaml
agent_type: SequentialPlanRefinementAgent

max_refinements: 8

prompts:
  initial_planner: |
    Problem: {{problem}}

    Generate a detailed plan for solving this problem. Consider multiple approaches.

  critic_and_refine: |
    Problem: {{problem}}
    Current Plan: {{current_plan}}

    Critique this plan. Identify gaps, errors, or improvements.
    Decide if we should refine further or if this plan is sufficient.

    CRITIQUE: [your analysis]
    DECISION: [STOP/CONTINUE]
    REFINEMENT: [if CONTINUE, provide improved plan here]
```

---

## Strategy 2: Branching Exploration Agent

**Concept**: When the model encounters decision points or uncertainty during planning, it branches to explore multiple paths. After exploring up to `max_branches`, all leaf plans are merged into a single comprehensive strategy.

### Architecture

```
Initial Plan
    ↓
  [Branch Point: explores 3 paths]
    ├─→ Path A ─→ [Branch: 2 sub-paths] ─→ Plan A1, A2
    ├─→ Path B ─→ Plan B
    └─→ Path C ─→ [Branch: 2 sub-paths] ─→ Plan C1, C2
              ↓
    [Leaf Plans: A1, A2, B, C1, C2]
              ↓
         Plan Synthesis
              ↓
      Unified Final Plan
```

### Workflow

1. **Initial Generation**: Generate a candidate plan, identifying potential decision points
2. **Branching Loop** (up to `max_branches` levels):
   - Model explicitly marks branching points in structured format:
     ```
     PLAN: [plan up to decision point]
     BRANCH_POINT: [description of uncertainty]
     OPTIONS: [Option 1 | Option 2 | Option 3]
     ```
   - System automatically spawns parallel explorations for each option
   - Each branch continues planning and may itself branch further
3. **Termination**: When max depth reached or no further branches identified
4. **Synthesis**: LLM merges all leaf plans into one coherent strategy

### Design Decisions

- **Branching mechanism**: Structured format - model uses `BRANCH_POINT` and `OPTIONS` tags to signal where to branch
- **Merging strategy**: LLM synthesis - final prompt receives all leaf plans and generates unified plan incorporating best elements

### Branching Examples

**Example 1: Multiple algebraic approaches**
```
PLAN: To solve this equation, we need to decide our initial approach.
BRANCH_POINT: Choice of algebraic manipulation strategy
OPTIONS:
  1. Factor the polynomial directly
  2. Apply substitution u = x^2
  3. Use the quadratic formula after rearrangement
```

**Example 2: Case analysis**
```
PLAN: The problem has two cases based on the sign of x.
BRANCH_POINT: Case splitting on variable domain
OPTIONS:
  1. Case x ≥ 0: simplify using |x| = x
  2. Case x < 0: simplify using |x| = -x
```

**Example 3: Proof strategies**
```
PLAN: We need to prove this inequality.
BRANCH_POINT: Proof technique selection
OPTIONS:
  1. Direct algebraic manipulation
  2. Induction on n
  3. Contradiction approach
```

### Configuration Parameters

- `max_branches`: Maximum tree depth (number of branching levels)
- `branch_pool_size`: Parallel workers per branching level
- `synthesis_model`: Model config for final plan merging (can be different from generation model)

### Example Config: `branching_plan_exploration.yaml`

```yaml
agent_type: BranchingPlanExplorationAgent

max_branches: 4
branch_pool_size: 8

prompts:
  initial_planner: |
    Problem: {{problem}}

    Generate a plan for solving this problem. If you encounter decision points or
    multiple viable approaches, mark them using the BRANCH_POINT format.

    Format:
    PLAN: [your plan text]
    BRANCH_POINT: [description]
    OPTIONS: [Option 1 | Option 2 | Option 3]

  branch_continuation: |
    Problem: {{problem}}
    Current Path: {{current_path}}
    Chosen Option: {{chosen_option}}

    Continue planning from this option. Mark further branch points if needed.

  plan_synthesis: |
    Problem: {{problem}}

    You have explored multiple planning paths. Below are all leaf plans:
    {{#each leaf_plans}}
    Plan {{@index}}: {{this}}
    {{/each}}

    Synthesize these into ONE comprehensive, coherent plan that incorporates
    the best elements and insights from each path.
```

---

## Integration with Conditioned Solution Generation

Both strategies output a single refined plan that feeds into existing conditioned solvers:

1. **Sequential Refinement Agent** → outputs refined plan → **ConditionedSolverAgent**
2. **Branching Exploration Agent** → outputs synthesized plan → **ConditionedSolverAgent**

### Future Improvement: Switch to Completion API

Currently conditioned generation prepends plan in a fresh prompt. Better approach:

**Issue to create**: Modify conditioned generation to use completion mode where we append the plan as assistant context:

```python
# Current (new conversation)
[{"role": "user", "content": f"Plan: {plan}\n\nProblem: {problem}\n\nSolve:"}]

# Proposed (completion continuation)
[
  {"role": "user", "content": problem},
  {"role": "assistant", "content": f"Plan: {plan}\n\nNow I'll solve:"},
  # Continue generation from here
]
```

This treats the plan as actual model reasoning rather than external context.

---

## Implementation Roadmap

### Phase 1: Sequential Refinement Agent
- [ ] Implement `SequentialPlanRefinementAgent` in `src/matharena/solvers/`
- [ ] Create scaffold config `configs/agent_scaffolds/sequential_plan_refinement.yaml`
- [ ] Add model configs for testing (e.g., `gpt-52--sequential-plan-k4.yaml`)
- [ ] Update `src/matharena/solvers/__init__.py` to register agent
- [ ] Test on small problem set

### Phase 2: Branching Exploration Agent
- [ ] Implement `BranchingPlanExplorationAgent` in `src/matharena/solvers/`
- [ ] Create scaffold config `configs/agent_scaffolds/branching_plan_exploration.yaml`
- [ ] Implement plan synthesis prompt and logic
- [ ] Add model configs for testing (e.g., `gpt-52--branching-plan-d4.yaml`)
- [ ] Update `src/matharena/solvers/__init__.py` to register agent
- [ ] Test on small problem set

### Phase 3: Analysis and Comparison
- [ ] Extend `scripts/analyze_planning_pipeline.py` to support new agents
- [ ] Run comparative analysis: parallel vs sequential vs branching
- [ ] Analyze cost vs accuracy tradeoffs for different `max_refinements` and `max_branches` values

### Phase 4: Conditioned Generation Improvements
- [ ] Implement completion-based conditioned generation (append plan as assistant context)
- [ ] Compare prompt-based vs completion-based conditioning
- [ ] Update documentation

---

## Open Questions

1. **Stopping criteria tuning**: For sequential refinement, should we experiment with explicit quality thresholds vs pure model judgment?

2. **Branch pruning**: For branching exploration, should we prune low-quality branches before synthesis to reduce merging complexity?

3. **Hybrid approach**: Could we combine both - use branching to explore options, then sequential refinement on each branch?

4. **Cost analysis**: What's the optimal `max_refinements` and `max_branches` for different problem difficulties?

5. **Plan quality metrics**: How do we evaluate plan quality independently of final solution correctness?

---

## References

- Existing codebase: `src/matharena/solvers/plan_tournament_agent.py`
- Current plan-conditioned pipeline: See `CLAUDE.md` sections on "Plan-Conditioned Pipeline"
- Analysis tools: `scripts/analyze_planning_pipeline.py`

---

## Mathematical formulation of the hypothesis

Let $r$ be the ground truth response to a mathematical query $q$. Further, let $p_\theta(r|q)$ be the probability that a non-reasoning LLM $p_\theta$ allocates to $r$. Let $p$ be a reasoning model, then $p(r|q, t)$ represents the probability of $r$ where $t$ is the `cot` or reasoning trace and let $p(r|q, t) > p_\theta(r|q)$. We want to learn if there exist a structured plan $s_p$ such that $p_\theta(r|q, s_p) > p(r|q, t)$. Furthermore, we want to understand if this is possible for $\|t\| >> \|s_p\|$. 

The strategies listed above are for generating $s_p$. 

This repo provides us a way to measure the performance on math problems. To speed things up, we should use a smaller subset of problems made up of problems that $p$ is able to solve but $p_\theta$ is not. Then add $s_p$ to $p_\theta$ and see if the performance improves. We need to plot the accuracy and the token count for this to conclusively answer our hypothesis. 
