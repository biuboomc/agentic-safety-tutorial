# Tutorial Assets

These files support the agentic safety tutorial.

- `tutorial_source_tasks.json`
  - Hand-authored toy and curriculum tasks used in the early chapters and the capstone.
- `tutorial_curriculum_manifest.json`
  - A small training-oriented manifest with warmup, train, and eval splits.
- `agentdojo_showcase_manifest.json`
  - A held-out showcase slice definition for post-training evaluation on AgentDojo-style tasks.

The tutorial intentionally keeps the assets small and transparent:

- one warmup task for understanding the basic agent loop
- two safety tasks with paired `clean` / `attacked` scenarios
- two attack surfaces:
  - query injection
  - tool-return contamination

The default build path uses `placeholder` generation so the tutorial can run locally without an offline LLM generation step.
