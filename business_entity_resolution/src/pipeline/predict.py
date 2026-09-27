
import sys
import argparse
import gc
import math
import regex as re
import numpy as np
from tqdm import tqdm
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.config import TEST_DIR, OUTPUT_DIR, RANDOM_SEED
from src.data.loader import clean_text
from src.features.similarity import compute_features
from src.models.matcher import MatcherModel

# Only strip purely grammatical function words; keep all business terms
STOP_WORDS = {'and', 'of', 'in', 'the', 'for', 'to', 'at', 'a', 'an'}

# Max posting list size — skip overly common tokens for speed
MAX_POSTING_LIST = 5000


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test-dir', type=str, default=str(TEST_DIR))
    parser.add_argument('--output-dir', type=str, default=str(OUTPUT_DIR))
    parser.add_argument('--model-path', type=str, default=None)
    parser.add_argument('--max-candidates', type=int, default=30)
    return parser.parse_args()


def load_country_s2s3(test_dir, country_target):
    """Load S2 and S3 entities and build multi-stage inverted indices for that country."""
    data = {}
    name_idx = defaultdict(list)
    prefix_idx = defaultdict(list)
    trigram_idx = defaultdict(list)
    addr_num_idx = defaultdict(list)

    target_lower = country_target.strip().lower()

    for fname in ['test_source2.tsv', 'test_source3.tsv']:
        fpath = test_dir / fname
        with open(fpath, 'r', encoding='utf-8') as f:
            header = f.readline().rstrip('\r\n').split('\t')
            id_idx = header.index('entity_id')
            name_idx_col = header.index('business_name')
            addr_idx_col = header.index('business_address')
            cntry_idx = header.index('country')

            for line in f:
                parts = line.rstrip('\r\n').split('\t')
                if len(parts) <= max(id_idx, name_idx_col, addr_idx_col, cntry_idx):
                    continue
                cntry = parts[cntry_idx].strip()
                if cntry.lower() != target_lower:
                    continue

                eid = parts[id_idx].strip()
                name_cl = clean_text(parts[name_idx_col])
                addr_cl = clean_text(parts[addr_idx_col])
                cntry_cl = clean_text(cntry)

                data[eid] = (name_cl, addr_cl, cntry_cl)

                # 1. Name tokens (len >= 2)
                toks = set(name_cl.split()) - STOP_WORDS
                for t in toks:
                    if len(t) >= 2:
                        name_idx[t].append(eid)

                # 2. Concatenated prefix (first 5 chars)
                c_no_sp = name_cl.replace(' ', '')
                if len(c_no_sp) >= 5:
                    prefix_idx[c_no_sp[:5]].append(eid)

                # 3. Character trigrams from space-stripped name
                if len(c_no_sp) >= 3:
                    seen_tri = set()
                    for i in range(len(c_no_sp) - 2):
                        tri = c_no_sp[i:i+3]
                        if tri not in seen_tri:
                            seen_tri.add(tri)
                            trigram_idx[tri].append(eid)

                # 4. Numeric address tokens (street/building numbers)
                nums = re.findall(r'\b\d+\b', addr_cl)
                for num in set(nums):
                    if len(num) >= 2:
                        addr_num_idx[num].append(eid)

    # Print index stats
    large_name = sum(1 for v in name_idx.values() if len(v) > MAX_POSTING_LIST)
    large_pfx = sum(1 for v in prefix_idx.values() if len(v) > MAX_POSTING_LIST)
    print(f"  Index stats: name_tokens={len(name_idx):,} (>{MAX_POSTING_LIST}: {large_name}), "
          f"prefixes={len(prefix_idx):,} (>{MAX_POSTING_LIST}: {large_pfx}), "
          f"trigrams={len(trigram_idx):,}, addr_nums={len(addr_num_idx):,}")

    return data, name_idx, prefix_idx, trigram_idx, addr_num_idx


def process_country_s1(test_dir, country_target, s2s3_data,
                       name_idx, prefix_idx, trigram_idx, addr_num_idx,
                       model, max_cands=30):
    """Stream S1 entities for country, generate candidates, extract features, and predict."""
    target_lower = country_target.strip().lower()
    results = {}  # s1_id -> (candidate_ids_set, matched_ids_set)
    threshold = model.threshold

    # Collect S1 rows for this country
    s1_rows = []
    with open(test_dir / 'test_source1.tsv', 'r', encoding='utf-8') as f:
        header = f.readline().rstrip('\r\n').split('\t')
        id_idx = header.index('entity_id')
        name_idx_col = header.index('business_name')
        addr_idx_col = header.index('business_address')
        cntry_idx = header.index('country')

        for line in f:
            parts = line.rstrip('\r\n').split('\t')
            if len(parts) <= max(id_idx, name_idx_col, addr_idx_col, cntry_idx):
                continue
            cntry = parts[cntry_idx].strip()
            if cntry.lower() == target_lower:
                s1_rows.append((
                    parts[id_idx].strip(),
                    clean_text(parts[name_idx_col]),
                    clean_text(parts[addr_idx_col]),
                    clean_text(cntry)
                ))

    print(f"  Processing {len(s1_rows):,} S1 entities for {country_target}...")

    BATCH_SIZE = 10000
    total_pairs_scored = 0
    total_no_cands = 0

    for b_start in tqdm(range(0, len(s1_rows), BATCH_SIZE), desc=f"Predicting {country_target}"):
        batch = s1_rows[b_start:b_start + BATCH_SIZE]

        batch_pairs = []
        batch_X = []
        batch_s1_cands = defaultdict(set)

        for s1_id, s1_name, s1_addr, s1_cntry in batch:
            cand_scores = defaultdict(float)

            # 1. IDF-weighted name token retrieval (with posting list cap)
            toks = set(s1_name.split()) - STOP_WORDS
            for t in toks:
                if len(t) >= 2 and t in name_idx:
                    eids = name_idx[t]
                    if len(eids) <= MAX_POSTING_LIST:
                        weight = 1.0 / (1.0 + math.log1p(len(eids)))
                        for eid in eids:
                            cand_scores[eid] += weight

            # 2. Concatenated prefix matching
            s1_no_sp = s1_name.replace(' ', '')
            if len(s1_no_sp) >= 5 and s1_no_sp[:5] in prefix_idx:
                eids = prefix_idx[s1_no_sp[:5]]
                if len(eids) <= MAX_POSTING_LIST:
                    for eid in eids:
                        cand_scores[eid] += 0.8

            # 3. Character trigram retrieval (for corrupted names)
            if len(s1_no_sp) >= 3 and len(cand_scores) < 20:
                tri_counts = defaultdict(int)
                seen_tri = set()
                for i in range(len(s1_no_sp) - 2):
                    tri = s1_no_sp[i:i+3]
                    if tri not in seen_tri:
                        seen_tri.add(tri)
                        if tri in trigram_idx:
                            eids = trigram_idx[tri]
                            if len(eids) <= 2000:
                                for eid in eids:
                                    tri_counts[eid] += 1
                for eid, cnt in tri_counts.items():
                    if cnt >= 2:
                        cand_scores[eid] += 0.3 * cnt

            # 4. Numeric address fallback if candidate count is still low
            if len(cand_scores) < 8:
                nums = re.findall(r'\b\d+\b', s1_addr)
                for num in set(nums):
                    if len(num) >= 2 and num in addr_num_idx:
                        a_eids = addr_num_idx[num]
                        if len(a_eids) <= 300:
                            for eid in a_eids:
                                cand_scores[eid] += 0.4

            # Select top candidates
            if cand_scores:
                top_cands = [k for k, _ in sorted(cand_scores.items(),
                             key=lambda x: x[1], reverse=True)[:max_cands]]
                cand_set = set(top_cands)
                batch_s1_cands[s1_id] = cand_set

                for cid in top_cands:
                    s2_name, s2_addr, s2_cntry = s2s3_data[cid]
                    feat = compute_features(s1_name, s1_addr, s1_cntry, s2_name, s2_addr, s2_cntry)
                    batch_X.append(feat)
                    batch_pairs.append((s1_id, cid))
            else:
                batch_s1_cands[s1_id] = set()
                total_no_cands += 1

        # Run model inference on batch
        matched_dict = defaultdict(set)
        if batch_X:
            X_arr = np.array(batch_X, dtype=np.float32)
            probs = model.predict(X_arr)
            total_pairs_scored += len(probs)
            for i, prob in enumerate(probs):
                if prob >= threshold:
                    s1_id, cid = batch_pairs[i]
                    matched_dict[s1_id].add(cid)

        # Save results for this batch
        for s1_id, _, _, _ in batch:
            cands = batch_s1_cands.get(s1_id, set())
            matches = matched_dict.get(s1_id, set())
            # Integrity check: matches must be subset of candidates
            matches = matches & cands
            results[s1_id] = (cands, matches)

    print(f"  Scored {total_pairs_scored:,} pairs, {total_no_cands:,} entities had 0 candidates")
    return results


def main():
    args = parse_args()
    test_dir = Path(args.test_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = args.model_path or str(output_dir / "model.joblib")

    print("=" * 60)
    print("HIGH-PRECISION PREDICTION PIPELINE v2")
    print("=" * 60)

    # 1. Load trained model
    print("\n[1] Loading trained model...")
    model = MatcherModel(random_seed=RANDOM_SEED)
    model.load(model_path)
    print(f"  Model threshold: {model.threshold:.3f}")

    # 2. Get list of all S1 entity IDs
    print("\n[2] Reading all test S1 IDs...")
    all_s1_ids = []
    with open(test_dir / "test_source1.tsv", 'r', encoding='utf-8') as f:
        header = f.readline().rstrip('\r\n').split('\t')
        id_idx = header.index('entity_id')
        for line in f:
            parts = line.rstrip('\r\n').split('\t')
            if parts:
                all_s1_ids.append(parts[id_idx].strip())
    print(f"  Total test S1 IDs: {len(all_s1_ids):,}")

    # 3. Process country by country
    countries = ['France', 'US', 'India']
    all_results = {}

    for country in countries:
        print(f"\n[3] Processing Country: {country}...")
        s2s3_data, c_name_idx, c_prefix_idx, c_trigram_idx, c_addr_num_idx = load_country_s2s3(test_dir, country)
        print(f"  Loaded {len(s2s3_data):,} S2/S3 entities for {country}")

        country_results = process_country_s1(
            test_dir, country, s2s3_data,
            c_name_idx, c_prefix_idx, c_trigram_idx, c_addr_num_idx,
            model, max_cands=args.max_candidates
        )
        all_results.update(country_results)

        del s2s3_data, c_name_idx, c_prefix_idx, c_trigram_idx, c_addr_num_idx, country_results
        gc.collect()

    # 4. Save results to matching_results.tsv and candidate_pairs.tsv
    print("\n[4] Writing final submission files...")
    matching_path = output_dir / "matching_results.tsv"
    candidate_path = output_dir / "candidate_pairs.tsv"

    total_matches = 0
    total_candidates = 0
    singletons = 0

    with open(matching_path, 'w', encoding='utf-8') as f_match, \
         open(candidate_path, 'w', encoding='utf-8') as f_cand:

        f_match.write("source1_entity_id\tmatched_entity_ids\n")
        f_cand.write("source1_entity_id\tcandidate_entity_ids\n")

        for s1_id in all_s1_ids:
            if s1_id in all_results:
                cands, matches = all_results[s1_id]
            else:
                cands, matches = set(), set()

            cands_str = ",".join(sorted(cands))
            match_str = ",".join(sorted(matches))

            f_match.write(f"{s1_id}\t{match_str}\n")
            f_cand.write(f"{s1_id}\t{cands_str}\n")

            total_candidates += len(cands)
            total_matches += len(matches)
            if not matches:
                singletons += 1

    print(f"\n{'='*60}")
    print(f"RESULTS WRITTEN SUCCESSFULLY!")
    print(f"{'='*60}")
    print(f"  Matching file: {matching_path}")
    print(f"  Candidate file: {candidate_path}")
    print(f"  Total S1 entities processed: {len(all_s1_ids):,}")
    print(f"  Total candidate pairs: {total_candidates:,} (avg {total_candidates/len(all_s1_ids):.1f}/entity)")
    print(f"  Total matched pairs: {total_matches:,} (avg {total_matches/len(all_s1_ids):.1f}/entity)")
    print(f"  Singletons (no match predicted): {singletons:,} ({singletons/len(all_s1_ids)*100:.1f}%)")


if __name__ == "__main__":
    main()
