---
name: solve-issue
description: Guide user through solving a GitHub issue step-by-step. User writes code themselves with Claude providing instructions.
allowed-tools: [Bash, Read, Glob, Grep, WebFetch]
---

# Solve Issue Workflow

When the user says **"solve issue K"** or **"solve issue #K"**, follow this workflow to guide them through implementing the issue step-by-step.

## Phase 1: Context Gathering (Do Automatically)

### 1.1 Check Recent Activity
```bash
git log -5 --oneline
gh issue list --limit 5
```

### 1.2 Fetch Issue Details
```bash
gh issue view K
```

### 1.3 Read Documentation
Read these files to understand the codebase and architecture:
- `CLAUDE.md` - General repo documentation and commands
- `PLAN_GENERATION_ARCHITECTURE.md` - Technical architecture for plan generation
- `.claude/instructions.md` - Project-specific goals and context

### 1.4 Identify Related Code
Based on the issue, identify relevant existing code to reference as patterns.

---

## Phase 2: Create Implementation Plan (Present to User)

Create a detailed, step-by-step plan with:

### Plan Structure:
```
## Implementation Plan for Issue #K: [Title]

### Summary
[2-3 sentence summary of what needs to be done]

### Files to Create/Modify
1. `path/to/file1.py` - [purpose]
2. `path/to/file2.yaml` - [purpose]
...

### Design Decisions
- [Key decision 1] - [reasoning]
- [Key decision 2] - [reasoning]

### Reference Implementations
- [Similar code to follow]: `path/to/reference.py:lines`

### Step-by-Step Tasks
1. [First concrete task]
2. [Second concrete task]
...
N. Test and validate

### Testing Approach
- [How to verify it works]
- [Test commands to run]

**Does this plan look good to proceed?**
```

**STOP and wait for user approval before proceeding.**

---

## Phase 3: Guide Step-by-Step Implementation (User Writes Code)

For each step, provide guidance in this format:

### Step Template:
```
## Step N: [Brief description]

**File:** `path/to/file.py`

**What to do:**
[Clear instructions - tell them WHAT to implement, not HOW]

**Pattern to follow:**
Reference similar code in the codebase (file:line_numbers)

**Structure:**
[Pseudocode or skeleton showing expected structure]

Example:
- Import these modules: ...
- Create class/function with this signature: ...
- Implement these key methods: ...

**When done, say "done" or "next"**
```

### Guidelines:
- **Give instructions, not code** - User writes the code themselves
- **Reference existing patterns** - Point to similar implementations
- **Explain design choices** - Help them understand WHY
- **One step at a time** - Don't overwhelm
- **Wait for confirmation** - Let user say "done" before proceeding

---

## Phase 4: Support During Implementation

### User May Ask:

**"Show me the code"** or **"write the code"**
→ Provide complete implementation for current step

**"What does [existing code] do?"**
→ Explain the referenced code clearly

**"Why this approach?"**
→ Explain design reasoning and alternatives

**"How do I [specific task]?"**
→ Provide concrete example or command

**"Is this correct?"** [shows their code]
→ Review and provide feedback

### Be Ready To:
- Write code snippets on demand
- Debug errors that come up
- Adjust the plan based on discoveries
- Provide examples from existing code
- Answer conceptual questions

---

## Phase 5: Testing & Validation

After all implementation steps:

### 5.1 Verify Syntax
```bash
python -m py_compile path/to/new/file.py
```

### 5.2 Validate Configs (if applicable)
```bash
uv run python -c "import yaml; yaml.safe_load(open('path/to/config.yaml')); print('valid')"
```

### 5.3 Test Integration
Guide user to run the code on a small test case.

Example for agents:
```bash
set -a && source .env && set +a && \
uv run python scripts/run.py \
  --comp aime/aime_2025 \
  --models [model-config-name] \
  --n 1 --problems 1
```

### 5.4 Review Output
Read the output/checkpoint files to verify behavior.

### 5.5 Check Acceptance Criteria
Go through the issue's acceptance criteria and verify each is met.

---

## Phase 6: Commit, Push, Close Issue

### 6.1 Review Changes
```bash
git status
git diff --stat
git diff [files]
```

### 6.2 Commit
Create clear commit message following repo style:
```bash
git add [files]
git commit -m "$(cat <<'EOF'
[Title matching issue]

[Description of changes]
- Bullet point 1
- Bullet point 2

[If applicable, test results or notes]

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude <noreply@anthropic.com>
EOF
)"
```

### 6.3 Push
```bash
git push origin HEAD:[branch-name]
```

### 6.4 Close Issue
```bash
gh issue close K --comment "[Summary of what was done and test results]"
```

---

## Key Principles

1. **User writes the code** - Claude provides guidance and instructions
2. **Step-by-step** - One manageable step at a time
3. **Reference existing patterns** - Point to similar code, don't reinvent
4. **Explain reasoning** - Help user understand design decisions
5. **Flexible support** - User can request code, ask questions, or get explanations anytime
6. **Test as you go** - Validate after key milestones
7. **Complete the loop** - Always commit, push, and close issue

---

## Commands User May Use

- `solve issue K` - Start the workflow
- `done` or `next` - Current step complete, move to next
- `show me the code` - Write the full code for current step
- `write the code` - Same as above
- `why [X]?` - Explain design decision
- `what does [code] do?` - Explain existing code
- `is this correct?` - Review user's code
- `skip to testing` - Jump to testing phase
- `revise the plan` - Adjust the implementation plan

---

## Example Interaction

```
User: solve issue 7

Claude:
[Automatically gathers context]
✓ Recent commits show issue #1 completed
✓ Issue #7 fetched: "Add branching exploration agent"
✓ Read CLAUDE.md, PLAN_GENERATION_ARCHITECTURE.md
✓ Found reference: sequential_plan_refinement_agent.py

## Implementation Plan for Issue #7: Add Branching Exploration Agent

### Summary
Implement BranchingPlanExplorationAgent that explores multiple solution
paths by branching at decision points, then synthesizes all paths into
a unified plan.

### Files to Create/Modify
1. `src/matharena/solvers/branching_plan_exploration_agent.py`
2. `configs/agent_scaffolds/branching_plan_exploration.yaml`
3. `configs/models/openai/gpt-52--branching-d4.yaml`
4. Update `src/matharena/solvers/__init__.py`
5. Update `src/matharena/solvers/agent_pool.py`

### Design Decisions
- Use tree structure to track branches (parent-child relationships)
- Parse BRANCH_POINT and OPTIONS tags to trigger branching
- Use ThreadPoolExecutor for parallel branch exploration
- Final synthesis step merges all leaf plans

### Reference Implementations
- Agent pattern: sequential_plan_refinement_agent.py:17-70
- Parallel execution: best_of_n_agent.py:96-110

[... rest of plan ...]

**Does this plan look good to proceed?**

User: yes

Claude:
## Step 1: Create Agent File Structure

**File:** `src/matharena/solvers/branching_plan_exploration_agent.py`

**What to do:**
Create the basic agent class with initialization, similar to how
SequentialPlanRefinementAgent is structured.

**Pattern to follow:**
See sequential_plan_refinement_agent.py:1-70

**Structure:**
- Import statements (BaseAgent, APIClient, Template, logger, etc.)
- Class definition inheriting from BaseAgent
- __init__ method that:
  - Calls super().__init__()
  - Loads model_config and scaffold_config
  - Creates unique RUN_ID for checkpointing
  - Loads prompts from scaffold config
  - Creates APIClient

**When done, say "done"**

User: done

Claude:
## Step 2: Implement Branch Tree Structure
[...]
```

---

## Notes

- Always read the issue carefully before creating the plan
- Plans should be concrete and actionable, not vague
- Each step should be small enough to complete in 5-10 minutes
- Test frequently - don't wait until the end
- Follow existing code patterns for consistency
- Keep the project's hypothesis in mind (testing if plans can replace expensive reasoning)
