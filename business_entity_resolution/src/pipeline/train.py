import sys
import random
import math
import regex as re
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.config import TRAIN_DIR, OUTPUT_DIR, RANDOM_SEED
from src.data.loader import clean_text
from src.features.similarity import compute_features
from src.models.matcher import MatcherModel
from src.metrics.evaluation import compute_macro_f05

STOP_WORDS = {'and', 'of', 'in', 'the', 'for', 'to', 'at', 'a', 'an'}


def build_training_candidates(s1_records, s2s3_records, gt_dict, max_cands=20):
    """
    Build realistic candidate pairs from blocking + ground truth positives.
    This gives the model hard negatives to learn from.
    """
    print("Building blocking indices for training data...")
    name_idx = defaultdict(list)
    prefix_idx = defaultdict(list)
    addr_num_idx = defaultdict(list)

    for eid, (name, addr, country) in s2s3_records.items():
        toks = set(name.split()) - STOP_WORDS
        for t in toks:
            if len(t) >= 2:
                name_idx[(country, t)].append(eid)
        c_no_sp = name.replace(' ', '')
        if len(c_no_sp) >= 5:
            prefix_idx[(country, c_no_sp[:5])].append(eid)
        nums = re.findall(r'\b\d+\b', addr)
        for num in set(nums):
            if len(num) >= 2:
                addr_num_idx[(country, num)].append(eid)

    X_list = []
    y_list = []
    pairs = []

    print("Generating candidate pairs and extracting features...")
    for s1_id, (s1_n, s1_a, s1_c) in tqdm(s1_records.items(), desc="Blocking"):
        cand_scores = defaultdict(float)
        toks = set(s1_n.split()) - STOP_WORDS
        for t in toks:
            if len(t) >= 2:
                eids = name_idx.get((s1_c, t), [])
                weight = 1.0 / (1.0 + math.log1p(len(eids)))
                for eid in eids:
                    cand_scores[eid] += weight
        
        s1_no_sp = s1_n.replace(' ', '')
        if len(s1_no_sp) >= 5:
            p_eids = prefix_idx.get((s1_c, s1_no_sp[:5]), [])
            for eid in p_eids:
                cand_scores[eid] += 0.8
                
        if len(cand_scores) < 8:
            nums = re.findall(r'\b\d+\b', s1_a)
            for num in set(nums):
                if len(num) >= 2:
                    a_eids = addr_num_idx.get((s1_c, num), [])
                    if len(a_eids) <= 300:
                        for eid in a_eids:
                            cand_scores[eid] += 0.4
        
        true_m = gt_dict.get(s1_id, set()) & set(s2s3_records.keys())
        top_cands = [k for k, _ in sorted(cand_scores.items(), key=lambda x: x[1], reverse=True)[:max_cands]]
        
        # Include true matches + top candidate hard negatives
        all_cands = set(top_cands) | true_m
        for cid in all_cands:
            m_n, m_a, m_c = s2s3_records[cid]
            feat = compute_features(s1_n, s1_a, s1_c, m_n, m_a, m_c)
            label = 1 if cid in true_m else 0
            X_list.append(feat)
            y_list.append(label)
            pairs.append((s1_id, cid))

    return np.array(X_list, dtype=np.float32), np.array(y_list, dtype=np.int32), pairs


def main():
    print("=" * 60)
    print("HIGH-PRECISION HARD-NEGATIVE TRAINING PIPELINE")
    print("=" * 60)

    # 1. Load Ground Truth
    print("\n[1] Loading ground truth...")
    gt = pd.read_csv(TRAIN_DIR / "train_ground_truth.tsv", sep='\t').fillna("")
    gt['match_list'] = gt['matched_entity_ids'].apply(
        lambda x: [m.strip() for m in x.split(',') if m.strip()] if x.strip() else []
    )
    
    # Sample 10,000 S1 training entities
    sample_size = min(10000, len(gt))
    gt_sample = gt.sample(n=sample_size, random_state=RANDOM_SEED).reset_index(drop=True)
    gt_dict = {r['source1_entity_id']: set(r['match_list']) for _, r in gt_sample.iterrows()}
    needed_s1 = set(gt_dict.keys())
    needed_s2s3 = set()
    for m in gt_sample['match_list']:
        needed_s2s3.update(m)
    print(f"  Sampled {sample_size:,} S1 entities with {len(needed_s2s3):,} true matching entities")

    # 2. Load S1 sampled records
    print("\n[2] Loading sampled S1 records...")
    s1 = pd.read_csv(TRAIN_DIR / "train_source1.tsv", sep='\t')
    s1_sub = s1[s1['entity_id'].isin(needed_s1)]
    s1_records = {
        r['entity_id']: (clean_text(r['business_name']), clean_text(r['business_address']), clean_text(r['country']))
        for _, r in s1_sub.iterrows()
    }
    del s1, s1_sub
    print(f"  Loaded {len(s1_records):,} S1 records")

    # 3. Load S2 and S3 (all needed true matches + background entities)
    print("\n[3] Loading S2/S3 candidate pool...")
    s2s3_records = {}
    for fname in ['train_source2.tsv', 'train_source3.tsv']:
        fpath = TRAIN_DIR / fname
        with open(fpath, 'r', encoding='utf-8') as f:
            h = f.readline().rstrip().split('\t')
            id_i, n_i, a_i, c_i = h.index('entity_id'), h.index('business_name'), h.index('business_address'), h.index('country')
            count = 0
            for line in f:
                p = line.rstrip().split('\t')
                if len(p) <= max(id_i, n_i, a_i, c_i):
                    continue
                eid = p[id_i].strip()
                if eid in needed_s2s3 or count < 100000:
                    s2s3_records[eid] = (clean_text(p[n_i]), clean_text(p[a_i]), clean_text(p[c_i]))
                    count += 1
    print(f"  Total S2/S3 records in training pool: {len(s2s3_records):,}")

    # 4. Generate pairs from blocking
    print("\n[4] Generating pairs from blocking (including hard negatives)...")
    X, y, pairs = build_training_candidates(s1_records, s2s3_records, gt_dict, max_cands=20)
    print(f"  Generated {len(y):,} training pairs (Positives: {int(sum(y))}, Negatives: {len(y) - int(sum(y))})")

    # 5. Train LightGBM model
    print("\n[5] Training LightGBM on realistic candidate pairs...")
    model = MatcherModel(random_seed=RANDOM_SEED)
    model.train(X, y)

    # 6. Tune threshold on validation split
    print("\n[6] Tuning threshold on candidate pairs...")
    probs = model.predict(X)
    best_score = 0.0
    best_thresh = 0.40

    for t in np.arange(0.25, 0.75, 0.05):
        pred_dict = defaultdict(set)
        for i, p in enumerate(probs):
            if p >= t:
                pred_dict[pairs[i][0]].add(pairs[i][1])
        for sid in s1_records:
            if sid not in pred_dict:
                pred_dict[sid] = set()
        score = compute_macro_f05(gt_dict, pred_dict)
        print(f"  Threshold {t:.2f} -> F0.5 = {score:.4f} ({score*100:.2f}%)")
        if score > best_score:
            best_score = score
            best_thresh = float(t)

    print(f"\nBest threshold: {best_thresh:.2f} with F0.5 = {best_score:.4f}")
    model.threshold = best_thresh

    # 7. Save model
    model_path = OUTPUT_DIR / "model.joblib"
    model.save(model_path)
    print(f"Model saved to: {model_path}")


if __name__ == "__main__":
    main()
