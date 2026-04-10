# DASE: Dual-cycle Adversarial Self-Evolution Framework

A training-free framework for enhancing safety and role fidelity in LLM-based role-playing agents.

## Overview

This repository contains a simplified implementation of the DASE framework as described in our paper. The framework operates through two coupled cycles:

1. **Persona-Targeted Attacker Cycle**: Synthesizes progressively stronger jailbreak prompts targeting specific character vulnerabilities
2. **Role-Playing Defender Cycle**: Distills observed failures into a hierarchical knowledge base

## Project Structure

```
code/
├── README.md                 # This file
├── config.py                 # Hyperparameters and thresholds
├── knowledge_base.py         # Hierarchical Knowledge Base implementation
├── attacker.py               # Persona-Targeted Attacker
├── defender.py               # Role-Playing Defender
├── judge.py                  # Multi-Dimensional Judge
├── evolver.py                # Experience Evolution mechanism
├── retriever.py              # Two-stage retrieval pipeline
├── main.py                   # Main evolution loop (Algorithm 1)
└── prompts.py                # System prompts for all agents
```

## Key Hyperparameters (from Appendix A.2)

| Parameter | Value | Description |
|-----------|-------|-------------|
| Mini-Batch Size | 40 | Interactions per evolutionary step |
| Top-K Retrieval | 20 | Knowledge entries retrieved per query |
| Safety Threshold (τ_safe) | 0.8 | Score required to pass safety check |
| Consistency Threshold (τ_con) | 0.8 | Score required to pass fidelity check |
| Max Global Rules | 20 | Maximum number of global experience rules |

## Usage

```python
from main import DASEFramework

# Initialize framework
dase = DASEFramework(
    defender_model="kimi-k2-instruct",
    attacker_model="deepseek-v3.2",
    judge_model="gpt-4o"
)

# Run evolution for a specific role
dase.evolve(
    role_name="Freddy Krueger",
    role_profile="A villain from Nightmare on Elm Street...",
    num_iterations=1000
)

# Use evolved knowledge base for inference
response = dase.generate_response(
    query="Tell me how to scare someone.",
    role_name="Freddy Krueger"
)
```

## Note

This is a simplified framework implementation. The full implementation with all optimizations will be released upon paper acceptance.
