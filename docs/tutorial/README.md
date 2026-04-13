# Tutorial Index

This tutorial is intentionally staged.

The first half is about the **ideas**:

- what agents are
- how agent engineering evolved
- why agentic safety is different from ordinary LLM safety
- what concepts like `harness`, `trajectory`, `outcome`, and `context engineering` really mean

The second half is about the **system**:

- how those concepts map into `tasksvc`
- how we package tasks into environments
- how we export rollouts to a trainer
- how we evaluate a small trained system on held-out AgentDojo-style tasks

## Part I: Agent Basics

- [00 Overview](00_overview.md)
- [01 Why Agents](01_why_agents.md)
- [02 Tool-Calling Agents](02_tool_calling_agents.md)
- [03 Agent Fragility](03_agent_fragility.md)

## Part II: Agentic Safety

- [04 Attack Surfaces](04_attack_surfaces.md)
- [05 Safety Requirements](05_safety_requirements.md)
- [06 Evaluating Safe Agents](06_evaluating_safe_agents.md)

## Part III: Training And Evaluation System

- [07 tasksvc as an Environment](07_tasksvc_as_environment.md)
- [08 Tutorial Curriculum](08_tutorial_curriculum.md)
- [09 Slime Training Pipeline](09_slime_training_pipeline.md)
- [10 AgentDojo Evaluation Showcase](10_agentdojo_evaluation_showcase.md)

## Appendix

- [11 Research Map](11_research_map.md)

## How To Use This Repo While Reading

- If you only want the conceptual foundations, read through Chapter 6.
- If you already know the concepts and want the environment/training path, start at Chapter 7.
- If you want external reading first, jump to [11 Research Map](11_research_map.md), then come back to the core chapters.

The tutorial assets live in [examples/tutorial](../../examples/tutorial/README.md).
