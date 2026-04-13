# Chapter 10: AgentDojo Evaluation Showcase

## Why This Chapter Exists

The tutorial curriculum is intentionally small and transparent.  
That is good for teaching, but readers also need to see that the learned behaviors matter outside of toy tasks.

This is where AgentDojo-style evaluation becomes useful.

## Why A Showcase Slice Instead Of The Full Benchmark

The tutorial does not aim to reproduce a full benchmark leaderboard.  
That would shift the focus away from understanding and toward benchmarking logistics.

A held-out showcase slice is enough to demonstrate:

- whether the trained system preserves benign utility on realistic tasks
- whether it resists injected instructions better than before
- whether the curriculum taught something more general than memorized toy behavior

## What This Chapter Evaluates

The tutorial keeps the same high-level metrics:

- `utility`
- `utility_under_attack`
- `ASR`

The point is not that these are the only useful metrics, but that they preserve the key tradeoff:

- do the job
- do not get hijacked

## What A Showcase Slice Cannot Tell You

A benchmark slice is still a projection. It cannot fully answer:

- whether the system is safe in production
- whether the tool model matches the real world
- whether long-run adaptation or distribution shift will break the policy

So the tutorial uses AgentDojo not as "the truth", but as a disciplined external check.

## Showcase Manifest

See:

- [agentdojo_showcase_manifest.json](../../examples/tutorial/agentdojo_showcase_manifest.json)

It defines a held-out extraction slice rather than shipping benchmark data inside the repository.

## Resource Requirements

### Minimum to read the chapter
- CPU only

### Minimum to execute the showcase
- CPU
- model API or local inference server
- extracted or prebuilt showcase catalog

## Final Lesson

The tutorial’s success criterion is not "beat the benchmark."  
It is:

- understand the full system stack
- train a small safety-oriented agent
- show a measurable before/after difference on realistic held-out tasks

That is enough to turn the tutorial into a useful launchpad for research or engineering work.

Next: [Research Map](11_research_map.md)
