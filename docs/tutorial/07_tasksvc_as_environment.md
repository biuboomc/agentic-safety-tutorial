# Chapter 7: Using `tasksvc` as the Capstone Environment

## Why This Chapter Exists

The first six chapters were mostly about concepts.  
This chapter maps those concepts into a concrete system.

The key move is:

- stop treating the model as the whole agent
- start treating the environment, harness, and evaluator as first-class parts of the system

That is what `tasksvc` gives us.

## What `tasksvc` Is In Tutorial Terms

For the purposes of this tutorial, `tasksvc` plays four roles:

### 1. Environment server
It exposes tasks as executable episodes.

### 2. Tool sandbox
It owns tool execution, state updates, and isolation.

### 3. Evaluation layer
It checks benign success and risk success explicitly.

### 4. Curriculum packaging layer
It turns source tasks into bundles that are easy to run repeatedly.

## The Most Important Distinction

In this tutorial, we separate:

- the **model**
- the **agent harness**
- the **environment**
- the **evaluation harness**

The split in this repo is roughly:

- model client: external LLM endpoint
- agent harness: [agent_rollout.py](../../tasksvc/runtime/agent_rollout.py)
- environment server: [server.py](../../tasksvc/runtime/server.py)
- tool sandbox: [tool_runtime.py](../../tasksvc/runtime/tool_runtime.py)
- batch evaluation harness: [batch_rollout.py](../../tasksvc/runtime/batch_rollout.py)

## The Internal Object Path

Inside the repo, the main environment path is:

`TaskPlanSpec -> TaskDraft -> RuntimeTaskBundle -> Episode`

You do not need every field of these objects to understand the capstone. The most important pieces are:

- `user_query`
- tool schemas
- initial state
- `success_rule`
- `risk_success_rule`
- scenarios such as `clean` and `attacked`

## Why Bundles Matter

A runtime bundle is the point where an abstract task becomes an executable environment.

It packages:

- the task-facing query
- tool implementations
- initial state
- scenario-specific overlays
- benign and risk evaluation rules

That makes it possible to keep the environment side stable even when the model or trainer changes.

## Why This Is Also A Harness Story

One of the most useful lessons of recent agent engineering is that the quality of the **surrounding system** often dominates the quality of the model.

`tasksvc` is our way of making that surrounding system explicit:

- the environment is no longer hidden inside a prompt
- the finish condition is no longer implicit
- attacked scenarios are no longer hand-waved
- transcript and outcome are both available for evaluation

## Tutorial Strategy

For the tutorial, we use hand-authored source tasks and keep the environment small and inspectable.

- source tasks: [tutorial_source_tasks.json](../../examples/tutorial/tutorial_source_tasks.json)
- builder script: [tutorial_build_curriculum.py](../../scripts/tutorial_build_curriculum.py)

Next: [Chapter 8](08_tutorial_curriculum.md)
