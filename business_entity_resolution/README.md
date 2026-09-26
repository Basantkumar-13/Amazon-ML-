# Business Entity Resolution Pipeline

This repository contains the end-to-end Machine Learning pipeline for the Amazon ML Challenge 2026.

## Structure
```
business_entity_resolution/
├── requirements.txt         # Pinned python environment dependencies
├── README.md                # Reproduction and execution guide
└── src/                     # Source code modules
    ├── config.py            # Global paths, seeds, thresholds, hyperparameters
    ├── data/                # Data loaders, cleaning, text normalization
    ├── blocking/            # Candidate generation & search space reduction
    ├── features/            # Feature extraction (string metrics, TF-IDF, lexical)
    ├── models/              # Matching classifiers & rankers (LightGBM/XGBoost)
    ├── metrics/             # Macro-averaged F_0.5 score computation
    └── pipeline/            # Train, inference, and submission generation scripts
```

## Reproduction Instructions

### 1. Environment Setup
Install the pinned requirements:
```bash
pip install -r requirements.txt
```

### 2. End-to-End Execution
Run the full pipeline to regenerate both candidate pairs and final predictions:
```bash
# 1. Feature extraction & model training (optional if pre-trained weights are cached)
python -m src.pipeline.train

# 2. Candidate generation & final prediction on test set
python -m src.pipeline.predict --test-dir ../../dataset/test --output-dir ../../output
```

### 3. Submission Validation
Validate the generated output before submission:
```bash
python ../../utils/validate_submission.py \
  --matching ../../output/matching_results.tsv \
  --candidate ../../output/candidate_pairs.tsv \
  --test-dir ../../dataset/test
```
