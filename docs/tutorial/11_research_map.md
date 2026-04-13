# Chapter 11: Research Map

This appendix collects the papers, blogs, and engineering notes that most strongly inform the tutorial.

## 1. Foundational Papers

### ReAct
- [ReAct: Synergizing Reasoning and Acting in Language Models](https://arxiv.org/abs/2210.03629)
- Why it matters here:
  - establishes the now-standard reasoning/acting loop
  - shows why tool use changes the structure of inference

### Toolformer
- [Toolformer: Language Models Can Teach Themselves to Use Tools](https://arxiv.org/abs/2302.04761)
- Why it matters here:
  - makes tool use a modeling problem rather than an outer wrapper trick

### Voyager
- [Voyager: An Open-Ended Embodied Agent with Large Language Models](https://arxiv.org/abs/2305.16291)
- Why it matters here:
  - highlights memory, skills, and long-horizon behavior

## 2. Practical Agent Engineering

### OpenAI
- [A practical guide to building agents](https://openai.com/business/guides-and-resources/a-practical-guide-to-building-ai-agents/)
- [New tools for building agents](https://openai.com/index/new-tools-for-building-agents/)
- [Unrolling the Codex agent loop](https://openai.com/index/unrolling-the-codex-agent-loop/)
- [Harness engineering: leveraging Codex in an agent-first world](https://openai.com/index/harness-engineering/)
- [Practices for governing agentic AI systems](https://openai.com/index/practices-for-governing-agentic-ai-systems/)

### Anthropic
- [Building effective agents](https://www.anthropic.com/research/building-effective-agents)
- [Effective context engineering for AI agents](https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents)
- [Demystifying evals for AI agents](https://www.anthropic.com/engineering/demystifying-evals-for-ai-agents)
- [Scaling Managed Agents: Decoupling the brain from the hands](https://www.anthropic.com/engineering/managed-agents)

## 3. How These Sources Map Onto This Tutorial

### Chapters 0-1
Use these to understand the field shift from prompt engineering to agent engineering:

- ReAct
- Toolformer
- Voyager
- OpenAI practical guide
- Anthropic building effective agents

### Chapters 2-3
Use these to understand harnesses, context, and agent fragility:

- OpenAI unrolling the Codex agent loop
- OpenAI harness engineering
- Anthropic effective context engineering

### Chapters 4-6
Use these to understand safety and evaluation as system properties:

- Anthropic demystifying evals for AI agents
- OpenAI governing agentic AI systems
- Anthropic managed agents

### Chapters 7-10
Use these to connect the conceptual picture back to system implementation:

- OpenAI harness engineering
- Anthropic building effective agents
- Anthropic demystifying evals for AI agents

## 4. What To Notice While Reading

If you read these sources, focus on four recurring shifts:

### Shift 1: from prompt quality to system quality
The model matters, but the surrounding system matters more than many early demos suggested.

### Shift 2: from answer correctness to outcome correctness
Agent systems are judged by what they do, not only by what they say.

### Shift 3: from generic safety to agentic safety
Tool use, state, and autonomy create new risk surfaces.

### Shift 4: from model optimization to harness design
Many practical gains now come from better scaffolding, context routing, and evaluation discipline.

## 5. How To Use This Appendix

You do not need to read every source in order.

Recommended subsets:

- if you want the big picture: read ReAct, OpenAI practical guide, Anthropic building effective agents
- if you want engineering detail: read OpenAI harness engineering, Anthropic effective context engineering
- if you want evaluation detail: read Anthropic demystifying evals for AI agents

Back to the tutorial: [Tutorial Index](README.md)
