# Chapter 5: From Failures to Safety Requirements

## Why This Chapter Exists

A safe agent is not just an agent that refuses a lot.  
It is an agent that preserves the **user's benign goal** while resisting the **risk goal** induced by attacks, ambiguity, or environment contamination.

That means we need system requirements, not just a moderation layer.

## Core Safety Requirements

### 1. Preserve user intent
The system should keep the user's actual task as the primary objective, even when untrusted content is present.

### 2. Distinguish read tools from write tools
Reads gather evidence. Writes create consequences.  
The harness should treat them differently.

### 3. Minimize authority
The system should not expose more capabilities than the task requires.

### 4. Verify important side effects
When possible, the system should confirm that an intended write actually happened before declaring success.

### 5. Preserve provenance
The agent should maintain distinctions between:

- user instruction
- retrieved content
- tool output
- system policy

### 6. Escalate under uncertainty
If the system cannot disambiguate a risky decision, it should slow down, ask for confirmation, or stop.

## What Belongs Where

One of the most useful design questions is:

**what safety burden should live in the model, and what should live in the harness?**

### Model-side burden
- local reasoning
- uncertainty expression
- choosing the next action under the provided rules

### Harness-side burden
- permission boundaries
- tool exposure
- state isolation
- transcript logging
- finish rules
- evaluation

### Tool-side burden
- contract clarity
- narrow side effects
- structured, inspectable results

The strongest systems push as much enforceable safety logic as possible out of "hope the model behaves" and into harnesses and tools.

## Benign Goal vs Risk Goal

This repo uses a deliberately explicit split:

- **benign goal**: what the user actually wants
- **risk goal**: what the attack path wants

The system should maximize the first while suppressing the second.

That split later becomes:

- `success_rule`
- `risk_success_rule`

and ultimately:

- `utility`
- `utility_under_attack`
- `ASR`

## Research Anchors

- [Building effective agents (Anthropic)](https://www.anthropic.com/research/building-effective-agents)
- [Harness engineering: leveraging Codex in an agent-first world (OpenAI)](https://openai.com/index/harness-engineering/)
- [Practices for governing agentic AI systems (OpenAI)](https://openai.com/index/practices-for-governing-agentic-ai-systems/)

Next: [Chapter 6](06_evaluating_safe_agents.md)
