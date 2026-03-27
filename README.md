# PEFT Under Limited Resources

A controlled empirical study on parameter-efficient fine-tuning (PEFT) methods under limited resources.

## Project Goal

This project builds a unified and reproducible comparison framework for several PEFT methods under similar trainable-parameter budgets.

The goal is not to invent a new PEFT algorithm, but to evaluate which PEFT method is more practical under limited-resource conditions.

## Research Question

Under limited trainable-parameter budgets, how do different PEFT methods compare in:

- performance
- stability
- efficiency

## Starter Scope

- **Base model:** one backbone only
- **Task:** text classification
- **Dataset:** SST-2
- **Methods:** LoRA, Adapter, Prefix-Tuning, Prompt-Tuning

## Evaluation Metrics

### Task Performance
- Accuracy
- F1 Score

### Training Behavior
- Training loss
- Validation loss
- Convergence trend

### Efficiency
- Trainable parameter count
- Total training time
- Peak memory usage

### Stability
- Mean and standard deviation across repeated runs

## Fairness Criteria

All formal comparisons should follow the same rules:

- same base model and tokenizer
- same dataset split
- same optimizer and learning-rate schedule as much as possible
- same evaluation metrics
- approximately matched trainable-parameter budgets
- all runs must save exact configurations for reproducibility

## Planned Workflow

Research Question → Scope Definition → Task/Dataset Selection → Unified PEFT Pipeline → Pilot Run → Parameter Alignment → Formal Experiments → Analysis → Final Report

## Project Structure

```text
project_root/
├── data/
├── src/
├── configs/
├── notebooks/
├── results/
├── report_notes/
├── scripts/
├── README.md
└── requirements.txt
