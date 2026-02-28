# Detecting Patronising and Condescending Language (PCL)

**Coursework -- Natural Language Processing**\
Leaderboard Name: **PPSTAR**

This project implements transformer-based models for detecting
Patronising and Condescending Language (PCL) following the official
SemEval 2022 Task 4 framework.

The system covers:

-   Task 1: Binary classification (PCL vs Non-PCL)
-   Task 2: Multi-label classification of rhetorical categories
-   Official dataset reconstruction
-   Exploratory Data Analysis (EDA)
-   Class imbalance handling
-   Threshold optimisation
-   Error analysis

------------------------------------------------------------------------

## 1. Repository Structure

    .
    ├── BestModel/
    │   └── nlp_cw.ipynb       # Main notebook (all experiments and final models)
    ├── dev.txt             # Task 1 development predictions
    ├── test.txt            # Task 1 test predictions
    └── README.md           # Project documentation

------------------------------------------------------------------------

## 2. Environment Setup

Python version used:

    Python 3.10+

Install required dependencies:

``` bash
pip install transformers datasets accelerate scikit-learn torch pandas matplotlib
```

GPU is recommended but not required.

------------------------------------------------------------------------

## 3. Dataset

The dataset is automatically downloaded within the notebook.

Files retrieved programmatically:

-   dontpatronizeme_pcl.tsv
-   Official SemEval train split
-   Official SemEval development split
-   Official SemEval test set
-   Multi-label category annotations

No manual download is required.

The notebook reconstructs the official train/dev split using the
provided `par_id` mappings to ensure exact alignment with the
competition setup.

------------------------------------------------------------------------

## 4. How to Reproduce Results

Open:

    nlp_cw.ipynb

Run all cells sequentially from top to bottom.

The notebook performs:

1.  Installation and imports\
2.  Official dataset reconstruction\
3.  Data preprocessing\
4.  Exploratory Data Analysis (EDA)\
5.  Task 1 training (binary classification)\
6.  Threshold optimisation on development set\
7.  Task 2 training (multi-label classification)\
8.  Per-label threshold tuning\
9.  Error analysis\
10. Generation of submission files

------------------------------------------------------------------------

## 5. Task 1 -- Binary Classification

Model: `roberta-base`\
Objective: Detect whether a paragraph contains PCL.

Training configuration:

-   Loss: Weighted CrossEntropyLoss\
-   Learning rate: 1.5e-5\
-   Batch size: 16\
-   Epochs: 4\
-   Weight decay: 0.01\
-   Evaluation strategy: per epoch\
-   Model selection based on development F1

Class imbalance is handled using automatically computed class weights.

Threshold optimisation is performed on the development set.

Final Development Results:

-   F1 (default threshold 0.5): 0.5971\
-   Best threshold: 0.20\
-   Best F1: 0.6125

------------------------------------------------------------------------

## 6. Task 2 -- Multi-Label Classification

Model: `roberta-base`\
Objective: Predict rhetorical subcategories of PCL.

Training configuration:

-   Loss: BCEWithLogitsLoss\
-   Label-specific `pos_weight` to address imbalance\
-   Learning rate: 1.5e-5\
-   Batch size: 16\
-   Epochs: 4

Per-label threshold optimisation is applied to improve macro F1.

Final Development Result:

-   Macro F1: 0.3841

Performance varies significantly across labels due to extreme imbalance.

------------------------------------------------------------------------

## 7. Generated Output Files

After running the notebook, the following submission files are
generated:

### Task 1

-   `dev.txt` (2093 lines)\
-   `test.txt` (3832 lines)

Each file contains one prediction per line (0 or 1), aligned with the
official data order.

These files correspond to the final model used for the reported results.

------------------------------------------------------------------------

## 8. Reproducibility Note

Model training uses stochastic optimisation (AdamW, shuffled
mini-batches).

While small numerical variations may occur across runs, the provided
`dev.txt` and `test.txt` files correspond exactly to the results
reported in `Report.pdf`(scientia).

To reproduce identical predictions, avoid re-training and instead use
the final generated outputs included in this repository.

------------------------------------------------------------------------

## 9. Key Observations

-   The dataset exhibits severe class imbalance (\~9.5% positive in Task
    1).\
-   Some Task 2 labels are extremely sparse, limiting achievable
    performance.\
-   Threshold tuning significantly improves F1 score in both tasks.\
-   Performance correlates strongly with label frequency.

------------------------------------------------------------------------

## 10. Leaderboard Information

Leaderboard name used for submission:

**PPSTAR**

------------------------------------------------------------------------

End of README.
