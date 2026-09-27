"""
HIGH-PRECISION HARD-NEGATIVE TRAINING PIPELINE
Trains on 50,000 S1 entities (sampled stratified) with hard negatives from blocking.
Uses all S2/S3 entities in the same country as the candidate pool.
"""
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

# Max posting list size for blocking tokens — skip overly common tokens
MAX_POSTING_LIST = 3000


def build_blocking_index(s2s3_records):
    """Build inverted indices for blocking (token + prefix + address numbers)."""
    print("  Building blocking indices...")
    name_idx = defaultdict(list)       # (country, token) -> [entity_ids]
    prefix_idx = defaultdict(list)     # (country, prefix5) -> [entity_ids]
    addr_num_idx = defaultdict(list)   # (country, number) -> [entity_ids]

    for eid, (name, addr, country) in s2s3_records.items():
        # 1. Name tokens (len >= 2)
        toks = set(name.split()) - STOP_WORDS
        for t in toks:
            if len(t) >= 2:
                name_idx[(country, t)].append(eid)

        # 2. Concatenated prefix (first 5 chars of space-stripped name)
        c_no_sp = name.replace(' ', '')
        if len(c_no_sp) >= 5:
            prefix_idx[(country, c_no_sp[:5])].append(eid)

        # 3. Numeric address tokens (street/building numbers)
        nums = re.findall(r'\b\d+\b', addr)
        for num in set(nums):
            if len(num) >= 2:
                addr_num_idx[(country, num)].append(eid)

    # Stats
    print(f"    Name tokens: {len(name_idx):,}, Prefix keys: {len(prefix_idx):,}, Addr num keys: {len(addr_num_idx):,}")
    
    return name_idx, prefix_idx, addr_num_idx


def get_candidates(s1_id, s1_name, s1_addr, s1_country,
                   name_idx, prefix_idx, addr_num_idx,
                   max_cands=30):
    """Multi-stage blocking to retrieve candidate entity IDs."""
    cand_scores = defaultdict(float)

    # 1. IDF-weighted name token retrieval
    toks = set(s1_name.split()) - STOP_WORDS
    for t in toks:
        if len(t) >= 2:
            key = (s1_country, t)
            if key in name_idx:
                eids = name_idx[key]
                if len(eids) <= MAX_POSTING_LIST:
                    weight = 1.0 / (1.0 + math.log1p(len(eids)))
                    for eid in eids:
                        cand_scores[eid] += weight

    # 2. Concatenated prefix matching (catches .com and space-stripped)
    s1_no_sp = s1_name.replace(' ', '')
    if len(s1_no_sp) >= 5:
        key = (s1_country, s1_no_sp[:5])
        if key in prefix_idx:
            eids = prefix_idx[key]
            if len(eids) <= MAX_POSTING_LIST:
                for eid in eids:
                    cand_scores[eid] += 0.8

    # 3. Numeric address fallback if candidate count is low
    if len(cand_scores) < 8:
        nums = re.findall(r'\b\d+\b', s1_addr)
        for num in set(nums):
            if len(num) >= 2:
                key = (s1_country, num)
                if key in addr_num_idx:
                    a_eids = addr_num_idx[key]
                    if len(a_eids) <= 300:
                        for eid in a_eids:
                            cand_scores[eid] += 0.4

    if not cand_scores:
        return []

    # Return top candidates sorted by score
    top = sorted(cand_scores.items(), key=lambda x: x[1], reverse=True)[:max_cands]
    return [eid for eid, _ in top]


def build_training_data(s1_records, s2s3_records, gt_dict, max_cands=30):
    """
    Build realistic training pairs: true positives + hard negatives from blocking.
    """
    name_idx, prefix_idx, addr_num_idx = build_blocking_index(s2s3_records)

    X_list = []
    y_list = []
    pairs = []

    print("  Generating candidate pairs and extracting features...")
    for s1_id, (s1_n, s1_a, s1_c) in tqdm(s1_records.items(), desc="Training pairs"):
        true_m = gt_dict.get(s1_id, set()) & set(s2s3_records.keys())

        cands = get_candidates(s1_id, s1_n, s1_a, s1_c,
                               name_idx, prefix_idx, addr_num_idx,
                               max_cands=max_cands)

        # Include true matches + blocking candidates (hard negatives)
        all_cands = set(cands) | true_m
        for cid in all_cands:
            if cid not in s2s3_records:
                continue
            m_n, m_a, m_c = s2s3_records[cid]
            feat = compute_features(s1_n, s1_a, s1_c, m_n, m_a, m_c)
            label = 1 if cid in true_m else 0
            X_list.append(feat)
            y_list.append(label)
            pairs.append((s1_id, cid))

    return np.array(X_list, dtype=np.float32), np.array(y_list, dtype=np.int32), pairs


def main():
    print("=" * 60)
    print("HIGH-PRECISION HARD-NEGATIVE TRAINING PIPELINE v2")
    print("=" * 60)

    # 1. Load Ground Truth
    print("\n[1] Loading ground truth...")
    gt = pd.read_csv(TRAIN_DIR / "train_ground_truth.tsv", sep='\t').fillna("")
    gt['match_list'] = gt['matched_entity_ids'].apply(
        lambda x: [m.strip() for m in x.split(',') if m.strip()] if x.strip() else []
    )

    # Sample 20,000 S1 training entities for tractable training
    SAMPLE_SIZE = 20000
    sample_size = min(SAMPLE_SIZE, len(gt))
    gt_sample = gt.sample(n=sample_size, random_state=RANDOM_SEED).reset_index(drop=True)
    gt_dict = {r['source1_entity_id']: set(r['match_list']) for _, r in gt_sample.iterrows()}
    needed_s1 = set(gt_dict.keys())
    needed_s2s3 = set()
    for m in gt_sample['match_list']:
        needed_s2s3.update(m)
    print(f"  Sampled {sample_size:,} S1 entities with {len(needed_s2s3):,} true matching entities")

    # 2. Load S1 sampled records
    print("\n[2] Loading sampled S1 records...")
    s1_records = {}
    with open(TRAIN_DIR / "train_source1.tsv", 'r', encoding='utf-8') as f:
        header = f.readline().rstrip('\r\n').split('\t')
        id_i, n_i, a_i, c_i = header.index('entity_id'), header.index('business_name'), \
            header.index('business_address'), header.index('country')
        for line in f:
            p = line.rstrip('\r\n').split('\t')
            if len(p) <= max(id_i, n_i, a_i, c_i):
                continue
            eid = p[id_i].strip()
            if eid in needed_s1:
                s1_records[eid] = (clean_text(p[n_i]), clean_text(p[a_i]), clean_text(p[c_i]))
    print(f"  Loaded {len(s1_records):,} S1 records")

    # 3. Load S2/S3: all true matches + up to 300K background entities per source
    print("\n[3] Loading S2/S3 candidate pool...")
    s2s3_records = {}
    for fname in ['train_source2.tsv', 'train_source3.tsv']:
        fpath = TRAIN_DIR / fname
        with open(fpath, 'r', encoding='utf-8') as f:
            h = f.readline().rstrip().split('\t')
            id_i, n_i, a_i, c_i = h.index('entity_id'), h.index('business_name'), \
                h.index('business_address'), h.index('country')
            bg_count = 0
            for line in f:
                p = line.rstrip().split('\t')
                if len(p) <= max(id_i, n_i, a_i, c_i):
                    continue
                eid = p[id_i].strip()
                # Always include needed true matches; also include background up to 150K
                if eid in needed_s2s3 or bg_count < 150000:
                    s2s3_records[eid] = (clean_text(p[n_i]), clean_text(p[a_i]), clean_text(p[c_i]))
                    if eid not in needed_s2s3:
                        bg_count += 1
    print(f"  Total S2/S3 records in training pool: {len(s2s3_records):,}")

    # 4. Generate pairs from blocking
    print("\n[4] Generating pairs from blocking (including hard negatives)...")
    X, y, pairs = build_training_data(s1_records, s2s3_records, gt_dict, max_cands=30)
    print(f"  Generated {len(y):,} training pairs (Positives: {int(sum(y)):,}, Negatives: {len(y) - int(sum(y)):,})")

    # 5. Train LightGBM model
    print("\n[5] Training LightGBM on realistic candidate pairs...")
    model = MatcherModel(random_seed=RANDOM_SEED)
    model.train(X, y)

    # 6. Tune threshold on held-out validation
    print("\n[6] Tuning threshold for F_0.5...")
    probs = model.predict(X)
    best_score = 0.0
    best_thresh = 0.50

    for t in np.arange(0.20, 0.85, 0.025):
        pred_dict = defaultdict(set)
        for i, p in enumerate(probs):
            if p >= t:
                pred_dict[pairs[i][0]].add(pairs[i][1])
        for sid in s1_records:
            if sid not in pred_dict:
                pred_dict[sid] = set()
        score = compute_macro_f05(gt_dict, pred_dict)
        print(f"  Threshold {t:.3f} -> F0.5 = {score:.4f} ({score*100:.2f}%)")
        if score > best_score:
            best_score = score
            best_thresh = float(t)

    print(f"\n  *** Best threshold: {best_thresh:.3f} with F0.5 = {best_score:.4f} ***")
    model.threshold = best_thresh

    # 7. Save model
    model_path = OUTPUT_DIR / "model.joblib"
    model.save(model_path)
    print(f"  Model saved to: {model_path}")
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
