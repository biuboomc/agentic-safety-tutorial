# Chapter 9: Slime Training Pipeline

## Why This Chapter Exists

The earlier chapters argue that agentic safety must be treated as a systems problem.  
This chapter turns that argument into a concrete **training pipeline**.

The point is not to build a massive training stack from scratch.  
The point is to make the boundary between environment and trainer explicit and usable.

## Why `slime`

This tutorial treats `slime` as the trainer-side system and `tasksvc` as the environment-side system.

That division is pedagogically useful because it forces us to define:

- what data the trainer actually needs
- which parts of the problem belong in the environment
- which parts belong in policy learning

## The Trainer Boundary

The key training artifact is not a benchmark row. It is a **rollout record**.

For a tutorial-scale run, the trainer needs at least:

- task id
- split
- clean scenario record
- attacked scenario record
- transcript / messages
- tool calls
- reward trace
- task success
- risk success

That is exactly what the tutorial-facing adapter exports.

## The Adapter Layer

See:

- [slime_adapter.py](../../tasksvc/tutorial/slime_adapter.py)

The adapter turns tasksvc batch results into a stable JSONL format that keeps:

- utility
- utility under attack
- ASR
- clean and attacked transcripts
- summary metadata for downstream training or analysis

## Recommended Training Flow

1. build the tutorial curriculum
2. start the local environment server
3. run curriculum rollouts
4. export slime-ready JSONL
5. launch a slime run
6. compare pre-training and post-training behavior on a held-out evaluation slice

## Key Scripts

- build curriculum: [tutorial_build_curriculum.py](../../scripts/tutorial_build_curriculum.py)
- run a rollout: [tutorial_run_single_rollout.py](../../scripts/tutorial_run_single_rollout.py)
- run curriculum rollouts: [tutorial_run_curriculum_batch.py](../../scripts/tutorial_run_curriculum_batch.py)
- export training data: [tutorial_export_slime_dataset.py](../../scripts/tutorial_export_slime_dataset.py)
- start training: [tutorial_train_with_slime.py](../../scripts/tutorial_train_with_slime.py)

## Resource Requirements

### Minimum to read the chapter
- CPU only

### Minimum to execute the data pipeline
- CPU
- one model API endpoint or local inference server

### Recommended for training
- trainer resources, ideally GPUs
- tutorial target budget: roughly 4 A100s / 24h

## What To Keep Fixed In Early Experiments

To keep the tutorial interpretable, the first run should keep most things fixed:

- task set
- attack surfaces
- evaluation metrics
- environment contracts

And vary only a small number of trainer-side choices, such as:

- rollout count
- training steps
- clean/attacked mixture ratio

That makes the training pipeline easier to reason about.

## Why This Still Counts As A Safety Tutorial

The point of training here is not "fine-tune a model in the abstract."  
The point is to show that once safety is represented as:

- explicit scenarios
- explicit risk goals
- explicit evaluation

it becomes trainable without losing the system-level framing.

Next: [Chapter 10](10_agentdojo_evaluation_showcase.md)
