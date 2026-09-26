# Methodology Document — Amazon ML Challenge 2026

## Team Name

Scrapper

## Problem Statement

Business Entity Resolution — given business records from 3 independent data sources with noisy fields, determine which records across sources refer to the same real-world business entity.

---

## Methodology Used

### Overall Approach

We use a classical entity resolution pipeline with 3 stages:

1. **Text Preprocessing** — normalize business names and addresses (lowercase, remove special chars, collapse whitespace)
2. **Blocking / Candidate Generation** — reduce the search space using an inverted index on country + name tokens
3. **Pairwise Classification** — extract similarity features for each candidate pair and classify using LightGBM

### Why This Approach

- Entity resolution at scale requires blocking to avoid O(n²) comparisons
- LightGBM is fast, handles tabular features well, and is MIT licensed
- F_0.5 rewards precision over recall, so we tune our threshold accordingly

---

## Candidate Generation / Blocking Strategy

### Method: Country + Token Inverted Index

For each S2/S3 entity, we index every word (length >= 2) in its cleaned business name, keyed by `(country, token)`.

For each S1 entity, we look up its name tokens in the index (filtered by same country) and collect all S2/S3 entities that share at least one token.

### Why This Works

- Country filter eliminates cross-country comparisons (businesses in US won't match India/France)
- Token overlap ensures at least one word in common — catches abbreviations, reorderings
- Handles the unseen country (France) automatically since we don't hardcode country values

### Trade-offs

- High recall (most true matches share at least one name token)
- May miss matches where names are completely different (rare in practice)
- Reduction ratio depends on vocabulary — common words like "the" generate more candidates

---

## Model Architecture and Feature Engineering

### Model: LightGBM Binary Classifier

- **Objective:** binary cross-entropy
- **Num leaves:** 31
- **Max depth:** 6
- **Learning rate:** 0.1
- **Early stopping:** 50 rounds on validation loss
- **License:** MIT (compliant with contest rules)

### Features (16 total)

| #   | Feature            | Description                                         |
| --- | ------------------ | --------------------------------------------------- |
| 1   | name_ratio         | Fuzzy ratio between business names                  |
| 2   | name_partial_ratio | Partial fuzzy ratio (handles substrings)            |
| 3   | name_token_sort    | Token-sorted fuzzy ratio (handles word reordering)  |
| 4   | name_token_set     | Token-set fuzzy ratio (handles extra/missing words) |
| 5   | name_jaccard       | Jaccard similarity on name word tokens              |
| 6   | name_levenshtein   | Normalized Levenshtein distance on names            |
| 7   | name_tfidf_cosine  | TF-IDF cosine similarity on names                   |
| 8   | addr_ratio         | Fuzzy ratio on addresses                            |
| 9   | addr_partial_ratio | Partial fuzzy ratio on addresses                    |
| 10  | addr_token_sort    | Token-sorted fuzzy ratio on addresses               |
| 11  | addr_jaccard       | Jaccard similarity on address tokens                |
| 12  | addr_levenshtein   | Normalized Levenshtein distance on addresses        |
| 13  | addr_tfidf_cosine  | TF-IDF cosine similarity on addresses               |
| 14  | country_match      | Binary: 1 if same country, 0 otherwise              |
| 15  | name_len_ratio     | min(len)/max(len) of business names                 |
| 16  | addr_len_ratio     | min(len)/max(len) of addresses                      |

### Feature Design Rationale

- **Fuzzy ratios** handle typos, abbreviations (Corp vs Corporation, Pvt vs Private)
- **Token-based metrics** handle word reordering and extra/missing words
- **TF-IDF cosine** weights rare words higher — discriminative business name tokens matter more
- **Levenshtein** captures character-level edit distance
- **Country match** is a strong prior — cross-country matches are very unlikely
- **Length ratios** help catch length mismatches (very different length = unlikely match)

---

## Training Procedure

1. **Sampling:** Due to memory constraints (16GB RAM), we sample 50K S1 entities from the 2.2M training set
2. **Pair Construction:** For each S1 entity, positive pairs come from ground truth; negative pairs are randomly sampled non-matching S2/S3 entities (3:1 negative-to-positive ratio)
3. **Train/Val Split:** 80/20 stratified split
4. **Training:** LightGBM with early stopping on validation loss
5. **Threshold Tuning:** We sweep thresholds from 0.3 to 0.85 and pick the one maximizing F_0.5 on the validation set

---

## Handling Singletons

- Singletons (S1 entities with no true matches) are correctly handled
- If blocking finds no candidates for an S1 entity, we output an empty matched_entity_ids
- This earns a full 1.0 F_0.5 score for that entity per the scoring rules

---

## Evaluation

- **Metric:** Macro-averaged F_0.5 (β = 0.5)
- F_0.5 = (1.25 × Precision × Recall) / (0.25 × Precision + Recall)
- Computed per S1 entity, then averaged across all S1 entities
- Precision is weighted 2× over recall

---

## Tools and Libraries Used

- **pandas** — data loading and manipulation
- **rapidfuzz** — fuzzy string matching (Levenshtein, ratios)
- **scikit-learn** — TF-IDF vectorizer, cosine similarity
- **LightGBM** — gradient boosted tree classifier
- **regex** — text normalization
- **numpy, scipy** — numerical operations
- **joblib** — model serialization
- **tqdm** — progress tracking

---

## Reproducibility

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Train the model
cd business_entity_resolution
python -m src.pipeline.train

# 3. Generate predictions
python -m src.pipeline.predict --test-dir ../dataset/test --output-dir ../output

# 4. Validate output
python ../utils/validate_submission.py \
  --matching ../output/matching_results.tsv \
  --candidate ../output/candidate_pairs.tsv \
  --test-dir ../dataset/test
```
