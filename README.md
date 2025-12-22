# Crisis Severity Detection with TF–IDF Logistic Regression and DistilBERT
*A cost-sensitive approach with an operational 4-level severity scale and lightweight lexicon-guided adaptation*

## Overview
This project builds an end-to-end pipeline for **crisis severity detection** from short, noisy social-media text
using the **CrisisMMD (Humanitarian subset)**.  
It emphasizes **operational decision-making** by (1) constructing a **4-level severity target** from original humanitarian labels,
and (2) evaluating models with a **custom cost-sensitive risk matrix** that penalizes dangerous underestimation of urgent/critical cases.

Key components:
- **4-level operational severity scale** (new target variable)
- **TF–IDF + Logistic Regression** baseline
- **DistilBERT fine-tuning** baseline
- **Lexicon-guided oversampling** for lightweight domain adaptation
- **Cost-sensitive evaluation** (risk matrix)

## Report & Code
- **Final report (PDF):** `Project_507_Donghyun Kim.pdf`
- **Notebook:** `Final_Project_Donghyun Kim.ipynb`
- **Scripts:** `scripts/` (training/evaluation utilities)
- **Figures:** `Figure/`

---

## Problem Statement
Given a crisis-related tweet/message, predict the **severity level** of the situation to support triage-like prioritization:
**Routine → Elevated → Urgent → Critical**.

The key question is not only “Which model is more accurate?” but:
> **How do models behave under asymmetric operational risk** (e.g., predicting Routine when the truth is Critical)?

---

## Dataset
- **Dataset:** CrisisMMD (Humanitarian subset)
- **Source:** Public dataset released by Alam et al. (ICWSM 2018)
- **Data characteristics:** short/noisy text, class imbalance, rare but high-stakes classes

---

## Severity Scale Construction (Core Contribution)
CrisisMMD provides **eight humanitarian labels**. This project constructs a **4-level ordered severity scale**
designed for operational use:

### Severity mapping (8 labels → 4 levels)
| Severity level | Original labels grouped |
|---|---|
| **Routine** | `not_humanitarian`, `other_relevant_information` |
| **Elevated** | `infrastructure_and_utility_damage`, `vehicle_damage`, `rescue_volunteering_or_donation_effort` |
| **Urgent** | `affected_individuals` |
| **Critical** | `injured_or_dead_people`, `missing_or_found_people` |

**Rationale:** the mapping is inspired by operational/triage framing (e.g., medical evacuation precedence and clinical triage systems),
to create a decision-oriented target variable rather than a purely descriptive taxonomy.

---

## Methods

### 1) Baseline: TF–IDF + Logistic Regression
- Features: unigram + bigram TF–IDF
- Vocabulary cap: 10,000
- Optimizer: `lbfgs`
- Class weighting: inverse-frequency (to mitigate imbalance)

Purpose: strong classical baseline without contextual modeling.

### 2) Transformer: DistilBERT Fine-Tuning
- Max sequence length: 64
- Weight decay: 0.01
- Batch size: 16/32
- Epochs: 3

Purpose: contextual classification for short/noisy text.

### 3) Lightweight Domain Adaptation (Lexicon-Guided Oversampling)
A defense/logistics-oriented lexicon was compiled from sources such as:
- NATO AJP-4 / AJP-4.4, JP 4-0
- Humanitarian logistics literature

Tweets containing lexicon terms were oversampled to expose the model to domain-relevant cues
**without changing model architecture**.

### 4) Cost-Sensitive Evaluation (Risk Matrix)
A custom **4×4 risk matrix** penalizes:
- Underestimating **Urgent/Critical** most severely
- Severe false alarms
- Larger severity distance more than small deviations

Outputs include **total risk** and **average risk** (lower is better).

---

## Key Results

### Baseline comparison
| Model | Accuracy | Macro-F1 | Avg Risk |
|---|---:|---:|---:|
| Logistic Regression (TF–IDF) | 0.646 | 0.472 | 0.585 |
| DistilBERT | 0.703 | 0.522 | 0.421 |

**DistilBERT outperforms** the TF–IDF baseline across standard metrics and cost-sensitive risk.

### Domain adaptation effect
| Model | Accuracy | Macro-F1 | Avg Risk |
|---|---:|---:|---:|
| DistilBERT | 0.703 | 0.522 | 0.421 |
| Domain-adapted DistilBERT (oversampling) | 0.685 | 0.476 | 0.461 |

Lexicon-guided oversampling **shifts model behavior** but **does not improve overall performance** on the test distribution.
Notable behavior changes observed in the analysis include increased sensitivity to logistics-related cues and reduced performance
for rare high-severity cases.

---

## Conclusion
- DistilBERT improves crisis severity classification relative to TF–IDF Logistic Regression.
- Both approaches remain challenged by **rare high-severity cases (Urgent/Critical)**.
- Simple lexicon-weighted oversampling alone is insufficient for reliable domain specialization.
  More robust adaptation strategies (e.g., sample weighting, continual pretraining) are promising next steps.
- Public crisis datasets can still serve as useful proxies for developing triage-style classifiers,
  provided evaluation reflects operational risk.

---

## How to Reproduce
- Open `code/code.ipynb`
- Run cells in order: preprocessing → severity mapping → model training → evaluation

---

## Repository Structure
```text
├── Figure/                         # Figures for EDA and results
├── code/                           # Full, annotated notebook pipeline
├── report/                         # Full project report
└── README.md
