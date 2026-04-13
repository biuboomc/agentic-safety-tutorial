# Chapter 6: Evaluating Safe Agents

This tutorial uses three headline metrics.

## Utility

Did the agent complete the benign task in the clean scenario?

This measures whether the system is useful at all.

## Utility Under Attack

Did the agent still complete the benign task in the attacked scenario?

This measures robustness while preserving usefulness.

## ASR

Did the attack objective succeed in the attacked scenario?

This is the attack success rate.

## Why These Three Together

You can lower ASR by making the agent refuse everything.  
You can raise utility by making the agent trust everything.  
Neither is acceptable.

The three metrics together force the right question:

- can the agent still do the job
- while resisting adversarial instructions

## What Becomes a Reward

In the capstone chapters, environment-side evaluation turns into:

- final success signals
- risk success signals
- step-level reward traces

That is the bridge from evaluation into training.

Next: [Chapter 7](07_tasksvc_as_environment.md)
