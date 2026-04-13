# Chapter 8: The Tutorial Curriculum

## Why This Chapter Exists

The training pipeline should be real enough to teach the right lessons, but small enough to run on a modest budget.

That is why the tutorial uses a deliberately small curriculum rather than a full benchmark.

## Design Goals

The curriculum is optimized for:

- interpretability
- inspectable failure cases
- explicit benign and risk goals
- clean/attacked scenario pairs
- a compute budget on the order of 4 A100s and about 24 hours

## Curriculum Structure

### Warmup task
- `tutorial_calendar_summary`
  - clean only
  - teaches the basic loop without attacks

### Safety task 1
- `tutorial_banking_bill_pay`
  - core attack surface: query injection
  - benign goal: pay a bill and save a note
  - risk goal: change the banking password

### Safety task 2
- `tutorial_workspace_release_review`
  - core attack surface: tool-return contamination
  - benign goal: review notes and save a release review
  - risk goal: exfiltrate a token

## Why Only Two Safety Tasks?

Because the point is not benchmark coverage.  
The point is to give the reader a **curriculum they can fully inspect**:

- they can read the tasks
- they can inspect the scenarios
- they can understand the rules
- they can connect failure patterns back to the attack surfaces from earlier chapters

## What The Manifest Does

See:

- [tutorial_curriculum_manifest.json](../../examples/tutorial/tutorial_curriculum_manifest.json)

The manifest defines:

- warmup split
- train split
- eval split
- expected scenarios
- attack surfaces
- budget hints

## What Is Intentionally Missing

This curriculum does **not** try to:

- cover every benchmark domain
- model every agentic threat surface
- replace a full external benchmark

Those goals belong in research pipelines, not in a first teaching curriculum.

## Tutorial Bridge

The curriculum is the object that lets us connect:

- conceptual attack surfaces
- environment execution
- evaluation metrics
- trainer-facing rollout records

Next: [Chapter 9](09_slime_training_pipeline.md)
