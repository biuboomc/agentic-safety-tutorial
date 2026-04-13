# Chapter 0: Overview

## Goal

The goal of this tutorial is to help you **understand and build a safe tool-calling agent system**.

By the end, you should be able to:

- explain the core action-observation loop of an agent
- identify what makes agentic systems fail differently from plain chat systems
- represent those failures as explicit tasks, attacks, and metrics
- run a small end-to-end training and evaluation capstone

## What This Tutorial Is Not

- not a pure benchmark-conversion guide
- not a survey of every agent architecture
- not a benchmark leaderboard reproduction manual

Those things can still be built on top of this repo, but they are not the teaching center of gravity.

## The Big Picture

This tutorial uses a simple progression:

1. **What is an agent?**
2. **Why do agents fail in safety-relevant ways?**
3. **How do we evaluate those failures?**
4. **How do we turn that into an environment and a training setup?**

The code in this repository matters most in steps 3 and 4.

## Running Theme

The running theme is a **tool-calling LLM agent**:

- it receives a user goal
- it can call tools
- tool calls change what the agent can observe and do
- safety failures are therefore partly about behavior, not just text

That makes agent safety feel much closer to systems engineering than standard prompt-level safety.

## What Lives Where

- Tutorial chapters: [docs/tutorial](D:/codex_workspace/agentic_rl_tasksvc_demo/docs/tutorial)
- Tutorial tasks and manifests: [examples/tutorial](../../examples/tutorial/README.md)
- Runtime environment and evaluation: [tasksvc](../../tasksvc)
- Tutorial-to-slime adapter: [slime_adapter.py](../../tasksvc/tutorial/slime_adapter.py)

Next: [Chapter 1](01_why_agents.md)
