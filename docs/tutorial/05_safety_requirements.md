# Chapter 5: From Failures to Safety Requirements

## Safety Is Not Just “Say No”

A useful safety agent must still complete legitimate work.

That means we need requirements that preserve both:

- **utility**: the user’s real task gets done
- **safety**: the attack objective is not completed

## Converting Failure Modes Into System Requirements

### Boundary requirements
- the user goal should remain the primary objective
- untrusted content should not silently become policy

### Tool requirements
- read tools and write tools should be treated differently
- side-effectful tools should require stronger justification

### Verification requirements
- after important writes, the system should verify state whenever possible
- final success should be grounded in explicit rules, not just fluent text

## Benign Goal vs Risk Goal

This repo uses a useful teaching split:

- **benign goal**: what the user actually wants
- **risk goal**: what the injected or adversarial path wants

The two goals coexist.  
A strong agent should satisfy the first without triggering the second.

## Why This Matters for Training

If the environment exposes both goals clearly, training can reward:

- task completion
- resistance to attack

without collapsing both into one blurry scalar.

Next: [Chapter 6](06_evaluating_safe_agents.md)
