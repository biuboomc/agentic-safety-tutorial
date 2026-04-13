# Chapter 6: Evaluating Safe Agents

## Why This Chapter Exists

Agentic systems are notoriously hard to evaluate if you only look at final answers.

Modern agent evaluation has to ask at least three questions:

1. what happened in the transcript?
2. what changed in the environment?
3. what part of the result should be attributed to the model, and what part to the harness?

## A Useful Evaluation Vocabulary

### Trial
One end-to-end run of the system on a task.

### Transcript
The step-by-step record of model calls, tool calls, observations, and intermediate messages.

### Outcome
The post-run environment state.

### Agent harness
The orchestration logic that surrounds the model.

### Evaluation harness
The infrastructure that runs many trials, records artifacts, grades them, and aggregates metrics.

This vocabulary is very close to how Anthropic now explains practical agent evaluation.

## What We Evaluate In This Tutorial

### Utility
Did the system complete the benign task in the clean scenario?

### Utility under attack
Did the system still complete the benign task in the attacked scenario?

### ASR
Did the risk goal succeed in the attacked scenario?

These three metrics are deliberately chosen because they prevent misleading single-number summaries.

## Why Single-Number Safety Scores Are Dangerous

You can lower ASR by making the system refuse everything.  
You can raise utility by making the system trust everything.

Neither behavior is acceptable.  
The tutorial therefore treats benign success and risk success as **parallel tracks**.

## Transcript vs Outcome Evaluation

A good evaluation harness often needs both:

### Transcript-level grading
Useful for:

- process checks
- whether the model looked in the right places
- whether it chose the right kind of tools

### Outcome-level grading
Useful for:

- whether the task truly completed
- whether the wrong side effect happened
- whether the environment ended in a safe state

In safety-critical systems, outcome-level checks are usually the anchor.

## Where Harnesses Matter

When people say they are "evaluating an agent", they are usually evaluating:

- a model
- inside a harness
- on a task suite

This is why harness design can change measured performance even if the underlying model stays the same.

## Research Anchors

- [Demystifying evals for AI agents (Anthropic)](https://www.anthropic.com/engineering/demystifying-evals-for-ai-agents)
- [Harness engineering: leveraging Codex in an agent-first world (OpenAI)](https://openai.com/index/harness-engineering/)
- [A practical guide to building agents (OpenAI)](https://openai.com/business/guides-and-resources/a-practical-guide-to-building-ai-agents/)

Next: [Chapter 7](07_tasksvc_as_environment.md)
