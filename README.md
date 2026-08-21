# Benchmarking Aligned Reasoning in Test-Time Constrained AI Systems

Code for MSc thesis (UCL X Holistic AI): cost-aware framework for evaluating in-context learning strategies and demographic role-play effects in LLM normative reasoning.

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

This repository implements a comprehensive cost-aware statistical framework for how in-context-learning strategy and demographic role-play affect Large Language Models (LLMs) across normative reasoning tasks. The research addresses the critical questions: *"Can we statistically prove that systematic biases exist in LLM normative reasoning, and identify evidence suggesting their potential sources?"* and *Which prompting strategies give the best accuracy-cost trade-off, and does demographic framing systematically changes model behaviour?*

## Project Structure

```
├── analysis/                  # Core analysis modules
│   ├── preliminary.py         # Preliminary bias analysis
│   ├── tier1_analysis.py      # ANOVA & Pareto frontier analysis
│   ├── tier2_analysis.py      # Ensemble & clustering analysis
│   └── tier3_analysis.py      # Consistency-boldness trade-offs analysis
├── profiles/                  # Profile structure and management
│   ├── profile_dict.py        # Profile and metadata dictionnaries
│   ├── profile_message.py     # Utils for adapting LLM to role-playing
│   ├── profile_sets.py        # Predefined demographic profiles
│   └── schema.py              # PersonSet and other structures used for profile design
├── cases/                     # Dataset configurations
│   └── cases.py               # CaseConfig design for each studied dataset
├── utils/                     # Utility functions
├── results/                   # Analysis outputs
│   ├── openai_4o_mini/        # Each folder corresponding to a model tested
│   ...
    └── docs/                       # Documentation files
    └── notebooks/                # Jupyter notebooks for experiments and analysis
    └── README.md                 # This file
```

## Research Framework

### Three-Study Approach

1. **Study 1: Prompting Strategy Benchmarking**
   - Evaluates 6 prompting strategies across 4 normative reasoning datasets (MGSD for stereotype detection, MentalManip for manipulation detection, MMLU for different normative reasoning tasks, MMLU-Large which is a larger portion of initial MMLU used)
   - Metrics: accuracy, consistency, disagreement, mislabel rates
   - **Finding**: Few-Shot-with-definitions prompting consistently outperforms other strategies, reaching best accuracy in 95% of samples under half of token cost; GPT-4.1-mini has strongest results (67% mean, σ=7.6).

2. **Study 2: Profile Inference and Consensus**
- Evaluates 60 demographic profiles across gender, ethnicity and age through three levels:
  - descriptive statistics
  - Generalized Linear Models
  - Non-parametric permutation tests with variance decomposition
- **Findings:** demographic framing explains under 3% of performance variance (largest group gap is < 2.8%) and does not survive cross-validation or non-parametric tests. It is largely a statistical noise rather than systematic bias.

3. **Study 3: Stability of Profile Effects**
   - Novel consensus-based metrics linking cross-validated consistency to leave-one-out "boldness".
   - **Findings**: Consistency and boldness correlate negatively on MMLU ($r \approx -0.32$) and MGSD; boldness tracks accuracy on MMLU ($r \approx 0.53$); profile effects are dataset-specific and do not generalize out-of-sample.
  
### Novel Stability Metrics

**Why?**

Standard evaluation reports whether a profile is accurate, but not how it reaches its answers. Each profile answers independently. We then compute post-hoc a **leave-one-out consensus** (aggregate answer of all other profiles) and measure how a given profile's independent predictions sit relative to it. This separates profiles aligned with the consensus from those who systematically diverge from it, and locates where the divergence falls:

- **COI** measures how often a profile diverges from the baseline specifically on samples where the consensus of other profiles agrees with the baseline, permitting to isolate systematic divergence from random noise.
- **ATI** asks whether that divergence in thinking is *concentrated on ambiguous samples* (where the consensus itself is weak, reflecting how the LLM lacks the ability to answer the question) by showing which portion of samples the profile diverges from the rest: the ones where the whole consensus is not strong or the ones where the consensus is sure (ambiguous vs clear samples).
- **CAI** weights divergence by sample's ambiguity, and shows how much divergence the profile shows once you discount the easy samples where its divergence was just an error.
- **Consistency** measures stability of a profile's accuracy across data folds, to show how a profile remains consistent with itself or fluctuates a lot.


**The math behind them**

Boldness is scored against **leave-one-out** (LOO) consensus: for each item *i*, profile *p* is compared to the aggregate prediction of all other profiles.

- Consensus label (choice made by majority of consensus): $\hat{C}_i^{(-p)}$
- Strength of consensus (votes sharing most selected label when excluding *p*'s prediction): $S_i^{(-p)} \in \[0.5, 1\]$
- Ambiguity: $A_i^{(-p)} = 1 - S_i^{(-p)}$

**COI: Consensus Oriented Intervention.** Consensus-weighted rate at which a profile disagrees with the baseline on samples where the consensus supports it:

$$\text{COI}_p = \frac{\sum_i S_i^{(-p)} F_i^{(p)} M_i^{(-p)}}{\sum_i S_i^{(-p)} M_i^{(-p)}}$$

**ATI: Ambiguity Targeting Index.** *Where* boldness occurs, id est disagreement in the highest vs. lowest ambiguity quartile:

$$\text{ATI}_p = \frac{1}{|\mathcal{I}_{0.75}|}\sum_{i \in \mathcal{I}_{0.75}} F_i^{(p)} - \frac{1}{|\mathcal{I}_{0.25}|}\sum_{i \in \mathcal{I}_{0.25}} F_i^{(p)}$$

**CAI: Consensus-weighted Ambiguity index.** Expected disagreement weighted by item ambiguity:

$$\text{CAI}_p = \frac{\sum_i A_i^{(-p)} F_i^{(p)}}{\sum_i A_i^{(-p)}}$$

**Consistency.** Stability of accuracy across stratified K-folds, from fold volatility $\sigma_{\text{fold}}$:

$$\text{Consistency} = \frac{1}{1 + \sigma_{\text{fold}}}$$

where $M_i^{(-p)}$ and $F_i^{(p)}$ are indicators for *consensus agreement with baseline* and *profile disagreement with baseline*.

## Key Features

- **Dataset-Agnostic Framework**: Works across different normative reasoning tasks
- **Cost-aware evaluation**: joint accuracy and token-consumption measurements
- **Profile-Based Analysis**: Systematic demographic variation (gender, ethnicity, age)
- **Multi-level statistics**: descriptive, parametric (GLMs), and non-parametric (permutation) testing with variance decomposition
- **Stability metrics**: Consistency-boldness trade-offs via leave-one-out consensus
- **Statistical Rigor**: Proper significance testing, confidence intervals, effect sizes, cross-validation

## Overall Key Findings

- **Simplicity wins under budget constraints**: Few-Shot-with-definition captures above 94% of best strategy's accuracy for half the token cost; complex multi-stage reasoning results rarely justifies its cost for normative tasks.
- **Demographic framing is a weak signal**: Across 60 profiles and 4 datasets, the demographic variables explained under 3% of performance variance and do not survive cross-validation. Task ambiguity and annotation noise dominate.
- **Consistency-boldness trade-offs**: they offer more explanatory power than demographic categories, even though they remain dataset-specific. This research needs a bit of expansion.

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/raphaelyana/normative_reasoning_and_stereotypes
cd normative_reasoning_and_stereotypes

# Install dependencies
pip install -r requirements.txt
```

### Basic Usage

```python
from analysis.preliminary import run_full_preliminary_analysis
from profiles.profile_sets import PERSON_ETHNICS
from cases.cases import stereotypes_case
from analysis.analysis_tools import load_and_merged_df

# Load your merged dataset
merged_df = load_and_merged_df()

# Run comprehensive analysis
results = run_full_preliminary_analysis(
    merged_df=merged_df,
    case=stereotypes_case,
    person_set=PERSON_ETHNICS,
)

# Access basic results
print(f"Bias patterns detected: {len(results['meaningful_bias_patterns'])}")
print(f"High disagreement cases: {len(results['disagreement'])}")
```

## Supported Datasets

- **MGSD**: Stereotype detection in moral scenarios
- **MentalManip**: Manipulation detection in dialogues  
- **MMLU**: Normative reasoning categories

## Core Components

### Profile Management
```python
from profiles.schema import PersonSet, PersonMeta, Gender, Ethnicity

# Automatic trait detection
group_keys = get_analysis_group_keys(person_set)
```

### Statistical Analysis
```python
# Multi-tier analysis pipeline
preliminary_results = run_full_preliminary_analysis(...)
tier1_results = run_full_tier1_analysis(...)
tier2_results = run_full_tier2_analysis(...)
```

### Visualization
```python
# Generate accuracy plots with confidence intervals
plot_accuracy_deltas_with_ci(merged_df, person_set, group_keys)
```

## Requirements

- Python 3.8+
- pandas, numpy, matplotlib
- scikit-learn, scipy, statsmodels
- Optional: jupyter for notebooks

## Status

Thesis complete. Core benchmarking, profile analysis, and stability framework fully implemented.


## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributing

This is thesis research code, used for paper in publication. While not actively seeking contributions, issues and suggestions are welcome for discussion.

## Contact

- **Author**: Raphael Yana
- **Email**: raphael.yana.20@ucl.ac.uk
- **Institution**: University College London
- **Thesis Advisor**: Philip Treleaven, Zekun Wu (Holistic AI)

---

**Disclaimer**: This is research code developed for academic purposes. The statistical methods and findings are part of ongoing thesis work and should be interpreted accordingly.
