"""
Configuration module for Business Entity Resolution pipeline.
Defines directory paths, hyperparameters, and operational constants.
"""

from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent
DATA_DIR = BASE_DIR / "dataset"
TRAIN_DIR = DATA_DIR / "train"
TEST_DIR = DATA_DIR / "test"
OUTPUT_DIR = BASE_DIR / "output"

# Output files
MATCHING_RESULTS_PATH = OUTPUT_DIR / "matching_results.tsv"
CANDIDATE_PAIRS_PATH = OUTPUT_DIR / "candidate_pairs.tsv"

# Metric constants
BETA = 0.5  # F_0.5 score weights precision 2x over recall

# Reproducibility
RANDOM_SEED = 42
