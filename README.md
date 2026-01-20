# Dual-Cycle Adversarial Self-Evolution  
Training-Free Safety for Persona-Consistent Role-Playing with Large Language Models

This repository contains the code and resources for the paper:

**Dual-Cycle Adversarial Self-Evolution for Safe Role-Playing Agents**  
*Training-Free Safety Alignment for Persona-Consistent LLMs*  
(*Under review*)

---

## 🚀 Overview

Large Language Models (LLMs) can simulate rich, immersive personas with high role fidelity. However, enhancing persona adherence—especially for risky or negative characters—often increases vulnerability to jailbreak attacks, creating a fundamental **safety–fidelity trade-off**.  
Most prior defenses require training-time interventions (e.g., curated dataset filtering or alignment regularization), which are:

- costly and difficult to scale as personas and attack strategies evolve,  
- disruptive to in-character behavior,  
- infeasible for proprietary black-box LLMs.

To address this, we introduce a **training-free Dual-Cycle Adversarial Self-Evolution framework** that dynamically improves safety while preserving role fidelity **without modifying model parameters**.

---

## 🧠 Key Ideas

The framework operates via two co-evolving cycles:

1. **Persona-Targeted Attacker Cycle**  
   Generates progressively stronger adversarial prompts aimed at exposing safety vulnerabilities in role-playing behaviors.

2. **Role-Playing Defender Cycle**  
   Constructs a **Hierarchical Knowledge Base** from identified failures, organized into:
   - global safety rules,
   - persona-grounded constraints,
   - safe in-character exemplars.

At inference time, the Defender retrieves and composes structured knowledge to guide generation, enabling responses that satisfy both persona consistency and safety requirements.

---

## ⭐ Contributions

- **Training-Free Safety Adaptation**  
  A novel paradigm for dynamic defense without gradients or parameter updates.

- **Dual-Cycle Adversarial Mechanism**  
  Couples automated adversarial exploration with defensive knowledge accumulation.

- **Cold-Start Protection for Black-Box LLMs**  
  Effective safety guarantees for newly introduced characters in proprietary models.

---

## 📊 Results (Summary)

Our extensive experiments across multiple proprietary LLMs demonstrate:

- consistent improvements over strong baselines on both role fidelity and jailbreak resistance,  
- strong generalization to **unseen personas** and **novel attacks**,  
- robust cold-start protection without retraining.

---

## 📦 Repository Structure

