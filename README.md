# Attention Economy and Information Intermediation Simulations

## Overview
This repository contains a collection of multi-agent simulation scripts designed to analyze the complex dynamics of the modern attention economy. The models mathematically explore the impact of information noise, algorithmic amplification, user fatigue, and institutional interventions on platform survival and sender payoffs.

## File Descriptions

* **`main1.py`**: Simulates the Dynamic Cheap Talk and Information Intermediation Model. It evaluates user strategies—platform dependence, autonomous fact-checking, and information avoidance—across varying noise variances and fact-checking costs through phase diagrams.
* **`main2.py`**: Implements a Heterogeneous Senders Model to demonstrate a dynamic "Market for Lemons". It compares the normalized attention gained by honest senders (Type-H) versus noise-makers (Type-L) and tracks how the ratio of noise-makers triggers attention collapse and impacts platform survival rates.
* **`main3.py` & `main4.py`**: Execute advanced simulations targeting specific modern platform challenges:
    * **Task 1**: Algorithmic Amplification and its impact on selective exposure and platform survival.
    * **Task 2**: Evolutionary Dynamics (Replicator Dynamics) showing strategic adaptation and audience capture between sender types over time.
    * **Task 3**: Institutional interventions, specifically testing the maximum tolerable delay and decay of Community Notes.

## Dependencies
The simulations rely on standard Python data science and plotting libraries:
- `numpy`
- `pandas`
- `matplotlib`

## Execution and Output
Run each script directly from your terminal or command prompt. For example:

```bash
python main3.py
