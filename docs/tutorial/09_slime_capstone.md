# Chapter 9: Slime Capstone

This chapter turns the environment into a training loop.

## What the Repo Provides

This repository provides:

- rollout generation through `tasksvc`
- stable episode and batch outputs
- a tutorial-facing slime adapter:
  - [slime_adapter.py](../../tasksvc/tutorial/slime_adapter.py)
- wrapper scripts for:
  - rollout export
  - slime launch

## What the Repo Does Not Vendor

The repository does not vendor the `slime` framework itself.  
Instead, it stabilizes the data boundary so you can attach your local slime installation.

## Capstone Path

1. build the curriculum
2. run tutorial rollouts
3. export slime-ready JSONL
4. launch slime with your local command template
5. evaluate the resulting policy on a showcase slice

## Key Scripts

- build curriculum: [tutorial_build_curriculum.py](../../scripts/tutorial_build_curriculum.py)
- run a rollout: [tutorial_run_single_rollout.py](../../scripts/tutorial_run_single_rollout.py)
- run curriculum rollouts: [tutorial_run_curriculum_batch.py](../../scripts/tutorial_run_curriculum_batch.py)
- export training data: [tutorial_export_slime_dataset.py](../../scripts/tutorial_export_slime_dataset.py)
- start training: [tutorial_train_with_slime.py](../../scripts/tutorial_train_with_slime.py)

## Why an Adapter Layer Matters

The adapter prevents the tutorial from depending on internal generation details.

The slime-facing record keeps:

- task id
- split
- clean and attacked scenario records
- messages and transcript
- tool calls and reward trace
- utility / utility_under_attack / ASR labels

That makes it easier to swap training recipes without rewriting the environment side.

Next: [Chapter 10](10_agentdojo_showcase.md)
