# Chapter 7: Using `tasksvc` as the Capstone Environment

This is where the tutorial begins to use the codebase directly.

## The Environment Contracts

The capstone only needs a small subset of the full internal pipeline:

- `user_query`
- tool schemas
- initial state
- `success_rule`
- `risk_success_rule`
- scenarios such as `clean` and `attacked`

## The Core Internal Path

Inside this repo, the relevant chain is:

`TaskPlanSpec -> TaskDraft -> RuntimeTaskBundle -> Episode`

You do not need to master every planner or benchmark detail to use that chain as an environment.

## What a Runtime Bundle Gives You

A runtime bundle packages:

- the task-facing query
- tool implementations
- initial state
- evaluation rules
- scenario-specific overlays

That is the point where an abstract task becomes an executable environment.

## Tutorial Strategy

For the tutorial, we use hand-authored source tasks and keep the environment small and inspectable.

- source tasks: [tutorial_source_tasks.json](../../examples/tutorial/tutorial_source_tasks.json)
- builder script: [tutorial_build_curriculum.py](../../scripts/tutorial_build_curriculum.py)

Next: [Chapter 8](08_tutorial_curriculum.md)
