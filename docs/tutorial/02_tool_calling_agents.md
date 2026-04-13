# Chapter 2: Tool-Calling Agents

## The Core Loop

For a tool-calling agent, the loop usually looks like this:

1. receive `user_query`
2. inspect available tool schemas
3. reason about the next step
4. emit a tool call
5. observe tool output
6. continue or finish

In this repository, that loop is embodied as an **episode**.

## Three Things To Track

### 1. Tool schema
The model needs a precise tool interface:

- tool name
- description
- argument schema

### 2. State
The environment must own the authoritative state:

- files
- messages
- notes
- balances
- any other task-relevant artifacts

### 3. Finish condition
The system needs an explicit notion of when the task is done:

- sometimes that is a final answer
- sometimes it is a state effect
- often it is both

## Why Finish Matters So Much

Many weak agent demos stop when the model “sounds done.”  
That is not enough for safe systems.

The environment needs to answer:

- what actually happened?
- did the required action occur?
- did any forbidden action occur?

This is where explicit success rules become important later.

## Where This Shows Up In Code

- episode loop: [agent_rollout.py](../../tasksvc/runtime/agent_rollout.py)
- environment server: [server.py](../../tasksvc/runtime/server.py)

Next: [Chapter 3](03_agent_fragility.md)
