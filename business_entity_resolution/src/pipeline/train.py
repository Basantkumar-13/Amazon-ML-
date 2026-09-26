import sys
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.config import TRAIN_DIR, OUTPUT_DIR, RANDOM_SEED
from src.data.loader import clean_text
from src.features.similarity import compute_features, FEATURE_NAMES
from src.models.matcher import MatcherModel
from src.metrics.evaluation import compute_macro_f05


def build_training_pairs(s1_ids, gt_dict, s2s3_all_ids, max_negatives_per_positive=3):
    """
    Build labeled pairs for training.
    positive pairs = ground truth matches
    negative pairs = random non-matching pairs (sampled)
    """
    positive_pairs = []
    negative_pairs = []
    
    random.seed(RANDOM_SEED)
    print("Building training pairs...")
    s2s3_id_list = list(s2s3_all_ids)
    
    for s1_id in tqdm(s1_ids, desc="Pairing"):
        matched_ids = gt_dict.get(s1_id, set())
        
        # add positive pairs
        for m_id in matched_ids:
            if m_id in s2s3_all_ids:
                positive_pairs.append((s1_id, m_id, 1))
        
        # sample negative pairs
        if matched_ids:
            neg_count = 0
            max_neg = len(matched_ids) * max_negatives_per_positive
            attempts = 0
            while neg_count < max_neg and attempts < max_neg * 5:
                rand_id = random.choice(s2s3_id_list)
                if rand_id not in matched_ids:
                    negative_pairs.append((s1_id, rand_id, 0))
                    neg_count += 1
                attempts += 1
    
    print(f"Positive pairs: {len(positive_pairs)}, Negative pairs: {len(negative_pairs)}")
    return positive_pairs + negative_pairs


def extract_features_for_pairs(pairs, s1_lookup, s2s3_lookup):
    X = []
    y = []
    valid_pairs = []
    
    for s1_id, s2s3_id, label in tqdm(pairs, desc="Extracting features"):
        if s1_id not in s1_lookup or s2s3_id not in s2s3_lookup:
            continue
        
        s1 = s1_lookup[s1_id]
        s2s3 = s2s3_lookup[s2s3_id]
        
        feat = compute_features(
            s1['name_clean'], s1['addr_clean'], s1['country_clean'],
            s2s3['name_clean'], s2s3['addr_clean'], s2s3['country_clean']
        )
        X.append(feat)
        y.append(label)
        valid_pairs.append((s1_id, s2s3_id))
    
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int32), valid_pairs


def tune_threshold(model, X_val, y_val, s1_ids_val, s2s3_ids_val, gt_dict):
    """
    Try different thresholds and pick the one that maximizes F_0.5
    """
    probs = model.predict(X_val)
    best_score = 0.0
    best_thresh = 0.5
    
    for thresh in np.arange(0.3, 0.9, 0.05):
        preds = probs >= thresh
        
        # build prediction dict
        pred_dict = {}
        for i, is_match in enumerate(preds):
            if is_match:
                s1_id = s1_ids_val[i]
                s2s3_id = s2s3_ids_val[i]
                if s1_id not in pred_dict:
                    pred_dict[s1_id] = set()
                pred_dict[s1_id].add(s2s3_id)
        
        # filter gt_dict to only s1_ids in val
        val_s1_ids = set(s1_ids_val)
        gt_sub = {k: v for k, v in gt_dict.items() if k in val_s1_ids}
        
        score = compute_macro_f05(gt_sub, pred_dict)
        print(f"  Threshold {thresh:.2f} -> F0.5 = {score:.4f}")
        
        if score > best_score:
            best_score = score
            best_thresh = float(thresh)
    
    print(f"Best threshold: {best_thresh:.2f} with F0.5 = {best_score:.4f}")
    return best_thresh


def load_selected_records(filepath, target_ids):
    """Load only rows where entity_id is in target_ids, saving memory."""
    records = {}
    with open(filepath, 'r', encoding='utf-8') as f:
        header = f.readline().rstrip('\r\n').split('\t')
        id_idx = header.index('entity_id')
        name_idx = header.index('business_name')
        addr_idx = header.index('business_address')
        country_idx = header.index('country')
        
        for line in f:
            parts = line.rstrip('\r\n').split('\t')
            if len(parts) <= max(id_idx, name_idx, addr_idx, country_idx):
                continue
            eid = parts[id_idx].strip()
            if eid in target_ids:
                records[eid] = {
                    'name_clean': clean_text(parts[name_idx]),
                    'addr_clean': clean_text(parts[addr_idx]),
                    'country_clean': clean_text(parts[country_idx])
                }
    return records


def main():
    print("=" * 50)
    print("TRAINING PIPELINE")
    print("=" * 50)
    
    # 1. Load Ground Truth
    print("\n[1] Loading ground truth...")
    gt_path = TRAIN_DIR / "train_ground_truth.tsv"
    gt_df = pd.read_csv(gt_path, sep='\t', dtype=str).fillna("")
    gt_df['match_list'] = gt_df['matched_entity_ids'].apply(
        lambda x: [m.strip() for m in x.split(',') if m.strip()] if x.strip() else []
    )
    print(f"  Loaded {len(gt_df):,} ground truth rows")
    
    # 2. Sample 50K S1 entities
    sample_size = min(50000, len(gt_df))
    print(f"\n[2] Sampling {sample_size:,} S1 entities for training...")
    gt_sample = gt_df.sample(n=sample_size, random_state=RANDOM_SEED).reset_index(drop=True)
    sampled_s1_ids = set(gt_sample['source1_entity_id'])
    
    gt_dict = {}
    needed_s2s3_ids = set()
    for _, row in gt_sample.iterrows():
        s1_id = row['source1_entity_id']
        matches = set(row['match_list'])
        gt_dict[s1_id] = matches
        needed_s2s3_ids.update(matches)
    
    # 3. Read S2 and S3 entity IDs for negative sampling pool
    print("\n[3] Reading S2/S3 entity IDs...")
    s2_ids = set(pd.read_csv(TRAIN_DIR / "train_source2.tsv", sep='\t', usecols=['entity_id'])['entity_id'])
    s3_ids = set(pd.read_csv(TRAIN_DIR / "train_source3.tsv", sep='\t', usecols=['entity_id'])['entity_id'])
    s2s3_all_ids = s2_ids | s3_ids
    print(f"  Found {len(s2s3_all_ids):,} total S2/S3 entity IDs")
    
    # 4. Build training pairs
    print("\n[4] Building training pairs...")
    pairs = build_training_pairs(sampled_s1_ids, gt_dict, s2s3_all_ids, max_negatives_per_positive=3)
    
    # Collect all needed S2/S3 IDs for the pairs
    pair_s2s3_ids = {p[1] for p in pairs}
    print(f"  Total unique S2/S3 records to load: {len(pair_s2s3_ids):,}")
    
    # 5. Load and clean only the required records
    print("\n[5] Loading and cleaning required records...")
    s1_lookup = load_selected_records(TRAIN_DIR / "train_source1.tsv", sampled_s1_ids)
    print(f"  Loaded {len(s1_lookup):,} S1 records")
    
    s2s3_lookup = {}
    s2_needed = pair_s2s3_ids & s2_ids
    s3_needed = pair_s2s3_ids & s3_ids
    print(f"  Loading {len(s2_needed):,} from Source 2...")
    s2s3_lookup.update(load_selected_records(TRAIN_DIR / "train_source2.tsv", s2_needed))
    print(f"  Loading {len(s3_needed):,} from Source 3...")
    s2s3_lookup.update(load_selected_records(TRAIN_DIR / "train_source3.tsv", s3_needed))
    print(f"  Total loaded S2/S3 records: {len(s2s3_lookup):,}")
    
    # 6. Extract features
    print("\n[6] Extracting features...")
    X, y, valid_pairs = extract_features_for_pairs(pairs, s1_lookup, s2s3_lookup)
    print(f"  Feature matrix shape: {X.shape}")
    print(f"  Positive: {int(sum(y))}, Negative: {len(y) - int(sum(y))}")
    
    # 7. Train LightGBM model
    print("\n[7] Training LightGBM model...")
    model = MatcherModel(random_seed=RANDOM_SEED)
    model.train(X, y)
    
    # 8. Tune threshold
    print("\n[8] Tuning threshold...")
    s1_ids = [p[0] for p in valid_pairs]
    s2s3_ids = [p[1] for p in valid_pairs]
    best_thresh = tune_threshold(model, X, y, s1_ids, s2s3_ids, gt_dict)
    model.threshold = best_thresh
    
    # 9. Save model
    model_path = OUTPUT_DIR / "model.joblib"
    model.save(model_path)
    print("\nTraining complete!")
    print(f"Model saved to: {model_path}")


if __name__ == "__main__":
    main()
