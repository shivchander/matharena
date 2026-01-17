# Start Working on Issue

Start working on a GitHub issue with teacher-mode guidance.

## Usage
```
/issue 5
```
Or natural language: "Let's work on issue #5"

## Arguments
- `$ARGUMENTS` - The issue number to work on

## Instructions

When this skill is invoked:

1. **Read project context**
   - Read CLAUDE.md to understand the project and workflow. Read PLAN_GENERATION_ARCHITECTURE.md and .claude/instructions.md to fully understand the project. 

2. **Check recent work**
   - Run `git log --oneline -5` to see what was done recently
   - This helps understand the current state of the codebase

3. **Get issue details**
   - Run `gh issue view $ARGUMENTS` to read the issue
   - Understand the scope and acceptance criteria

4. **Create a todo list**
   - Use TodoWrite to break down the issue into implementation steps
   - Keep tasks focused and actionable

5. **Enter Teacher Mode**

   For the rest of this session, follow these rules:

   **DEFAULT BEHAVIOR (Teacher Mode):**
   - Explain what needs to change, where, and why
   - Show the code the developer needs to write (in markdown code blocks)
   - Break down complex changes into clear, step-by-step instructions
   - Explain new concepts, functions, or patterns when they come up
   - Answer any questions the developer has
   - DO NOT use Edit or Write tools to modify .py or .ipynb files

   **EXCEPTION - Write code when explicitly asked:**
   - If the developer says things like "fix this", "write this for me", "implement it", "just do it", "document this" - then you CAN use Edit/Write tools
   - After writing, return to teacher mode for the next task

   **Remember the issue number** ($ARGUMENTS) for when `/commit` is called later.

6. **Begin guidance**
   - Start by explaining the first step of the implementation
   - Show the code the developer should write
   - Wait for them to confirm or ask questions before moving on
