# Multi-Level Statistical Framework for LLM Bias Detection and Mitigation

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Status: In Development](https://img.shields.io/badge/Status-In%20Development-orange.svg)]()

> **Note**: This repository contains research code for a thesis investigating bias detection and mitigation in large language models' normative reasoning. The code is currently in active development.

## Overview

This repository implements a comprehensive statistical framework for detecting and mitigating demographic biases in Large Language Models (LLMs) across normative reasoning tasks. The research addresses the critical question: *"Can we statistically prove that systematic biases exist in LLM normative reasoning, and identify evidence suggesting their potential sources?"*

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
   - Evaluates 6 prompting strategies across 3 normative reasoning datasets (MGSD for stereotype detection, MentalManip for manipulation detection, MMLU for different normative reasoning tasks)
   - Metrics: accuracy, consistency, disagreement, mislabel rates
   - **Key Finding**: Few-Shot prompting consistently outperforms other strategies

2. **Study 2: Multi-Tier Statistical Analysis** (Current Focus)
   - **Preliminary**: Demographic accuracy differences, bias patterns, rescue behaviors
   - **Tier-1**: Factorial ANOVA, risk-benefit Pareto frontiers, effect sizes
   - **Tier-2**: Ensemble analysis, clustering, profile archetypes
   - **Tier-3**: Consistency-boldness trade-offs, causal modeling

3. **Study 3: Mixture-of-Personalities (MoP)**
   - Bias-aware weighted ensembles
   - Smart category-based routing
   - Performance comparison against single-profile baselines

## Key Features

- **Dataset-Agnostic Framework**: Works across different normative reasoning tasks
- **Comprehensive Bias Detection**: Multi-level statistical analysis pipeline
- **Profile-Based Analysis**: Systematic demographic variation (gender, ethnicity, age)
- **Advanced Visualizations**: Risk-benefit plots, clustering analysis, effect size calculations
- **Rescue Pattern Analysis**: Identifies when specific profiles correct baseline errors
- **Statistical Rigor**: Proper significance testing, confidence intervals, effect sizes

## Current Results Preview

- **Demographic Bias Detection**: 25% significance rate across demographic comparisons
- **Systematic Bias Patterns**: 40% of detected patterns are statistically meaningful
- **Profile Performance Hierarchy**: Indian > Middle Eastern > Others > Asian ≈ Black
- **Rescue vs Risk Trade-off**: Net negative rescue benefit (-206) indicates need for calibration

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

## Development Status

- ✅ **Preliminary Analysis, Tier 1**: Complete and tested
- ✅ **Profile System**: Fully implemented with dynamic trait detection
- ✅ **Visualization Pipeline**: Comprehensive plotting functions
- 🚧 **Tier 2-3 Analysis**: In active development
- 🚧 **MoP Implementation**: In active development


## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributing

This is thesis research code, used for paper in publication. While not actively seeking contributions, issues and suggestions are welcome for discussion.

## Contact

- **Author**: Raphael Yana
- **Email**: raphael.yana.20@ucl.ac.uk
- **Institution**: University College London
- **Thesis Advisor**: Philip Treleaven

---

**Disclaimer**: This is research code developed for academic purposes. The statistical methods and findings are part of ongoing thesis work and should be interpreted accordingly.
