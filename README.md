# Anticancer-Peptide-Prediction
Interpretable Ensemble Learning for Anticancer Peptide Prediction via Physicochemical Property Integration
# Interpretable Ensemble Learning for Anticancer Peptide Prediction via Physicochemical Property Integration

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![Framework](https://img.shields.io/badge/Framework-Scikit--Learn%20%7C%20XGBoost-orange)](https://scikit-learn.org/)

## 📖 Introduction
This repository contains the source code and data for the paper **"Interpretable Ensemble Learning for Anticancer Peptide Prediction via Physicochemical Property Integration"**.

Anticancer peptides (ACPs) are promising therapeutic agents, but identifying them experimentally is costly. Existing deep learning methods often suffer from a lack of interpretability. In this study, we propose a novel **Ensemble Learning Framework** that explicitly integrates **Physicochemical Properties** (Charge & Hydrophobicity) with multi-view sequence features (AAC, CKSAAP).

The model achieves state-of-the-art performance on the ACP-2.0 dataset while providing clear biological insights into the mechanism of ACPs.

## ✨ Key Features
* **Multi-View Feature Extraction**: Combines Amino Acid Composition (AAC), Sequence Order (CKSAAP), and explicit **Physicochemical Properties** (Hydrophobicity & Charge).
* **Interpretable Feature Selection**: Uses an XGBoost-based selector to reduce dimensions from 822 to 262, filtering out noise.
* **Soft-Voting Ensemble**: Integrates **Random Forest (Bagging)** and **XGBoost (Boosting)** to balance bias and variance.
* **Biological Insight**: explicitly validates the "Amphipathic Mechanism" (Charge & Hydrophobicity are identified as top determinants).

## 📊 Performance
Benchmarked on the independent test set of **ACP-2.0**:

| Metric | Score |
| :--- | :--- |
| **Accuracy** | **77.54%** |
| **AUC** | **0.850** |
| **Specificity** | 79.7% |
| **Sensitivity** | 75.4% |
| **MCC** | 0.551 |

> **Comparison**: Our model outperforms the baseline **AntiCP 2.0 (75.58%)** and is comparable to large protein language models like **iACP-SEI (77.78%)**, but with significantly higher interpretability.

## 🛠️ System Architecture
The workflow consists of four stages:
1.  **Feature Extraction**: AAC + CKSAAP + Physicochemical Props.
2.  **Preprocessing**: Z-score Standardization.
3.  **Selection**: XGBoost-based thresholding (>1.2*mean).
4.  **Prediction**: Weighted Soft Voting (RF + XGB).

*(`![Workflow](images/Figure1.png)`)*

## 🚀 Quick Start

### 1. Prerequisites
Ensure you have Python 3.8+ installed. Install the dependencies:

```bash
pip install numpy pandas scikit-learn xgboost biopython matplotlib seaborn
```

### 2. Dataset Preparation
The model requires two FASTA format files placed in the project root directory:

* **`positive.fasta`**: Contains experimentally validated Anticancer Peptides (ACPs).
* **`negative.fasta`**: Contains Non-ACPs (negative samples).

> **Note**: The dataset used in this study is **ACP-2.0**. You can download the standard benchmark dataset from the [AntiCP 2.0 website](https://webs.iiitd.edu.in/raghava/anticp2/) or use your own custom datasets.

### 3. Usage
Once the dependencies are installed and data is prepared, you can run the complete workflow with a single command.

```bash
python main.py
```
### 4. Output & Visualization
The script performs the following tasks automatically:

1.  **Console Output**:

* **`Model Performance`**: Prints Accuracy, AUC, Sensitivity, Specificity, and MCC.

* **`Feature Ranking`**: Lists the Top 15 most important features (e.g., Avg_Charge, Lysine).

* **`Classification Report`**: Detailed precision/recall metrics for both classes.

2.  **Generated Figures (Saved in current directory)**:

* **`Figure2_Combined_CM.png`**: The confusion matrix comparison for Random Forest, XGBoost, and the Ensemble model.

* **`ROC Curve Plot`**: A visual comparison of ROC curves showing the Ensemble model's superiority (AUC = 0.850).

## 📂 Project Structure

```text
├── main.py              # Main training and evaluation script (Source code)
├── positive.fasta       # Positive samples (ACP-2.0)
├── negative.fasta       # Negative samples (ACP-2.0)
├── Figure2_Combined_CM.png  # Generated Confusion Matrix
├── README.md            # Project documentation
└── requirements.txt     # (Optional) Python dependencies
```

## 🧬 Biological Interpretability
A core contribution of this work is explaining why a peptide is anticancer. Our feature importance analysis revealed:

Electrostatic Interactions: Lysine (K) and Average Charge are the top-ranked features. This confirms that cationic properties are crucial for the initial attraction to negatively charged cancer cell membranes.

Hydrophobic Interactions: Leucine (L) and Average Hydrophobicity are highly ranked, validating the necessity of hydrophobic residues for membrane penetration.

These findings align with the "Amphipathic Mechanism" of anticancer peptides.
