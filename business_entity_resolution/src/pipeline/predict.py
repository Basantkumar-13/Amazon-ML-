import sys
import argparse
import gc
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from collections import defaultdict, Counter

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.config import TEST_DIR, OUTPUT_DIR, RANDOM_SEED
from src.data.loader import clean_text
from src.features.similarity import compute_features
from src.models.matcher import MatcherModel

# Common entity designators and noise words that do not provide discriminative blocking signal
STOP_WORDS = {
    'limited', 'private', 'llc', 'inc', 'ltd', 'pvt', 'corp', 'corporation',
    'company', 'co', 'llp', 'and', 'of', 'in', 'the', 'for', 'to', 'at', 'a', 'an',
    'sa', 'sarl', 'sas', 'gmbh', 'services', 'group', 'holdings', 'enterprises',
    'solutions', 'center', 'centre', 'care', 'health'
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test-dir', type=str, default=str(TEST_DIR))
    parser.add_argument('--output-dir', type=str, default=str(OUTPUT_DIR))
    parser.add_argument('--model-path', type=str, default=None)
    parser.add_argument('--max-candidates', type=int, default=15)
    return parser.parse_args()


def load_country_s2s3(test_dir, country_target):
    """Load S2 and S3 entities belonging to a specific country."""
    data = {}
    token_index = defaultdict(list)
    target_lower = country_target.strip().lower()
    
    for fname in ['test_source2.tsv', 'test_source3.tsv']:
        fpath = test_dir / fname
        with open(fpath, 'r', encoding='utf-8') as f:
            header = f.readline().rstrip('\r\n').split('\t')
            id_idx = header.index('entity_id')
            name_idx = header.index('business_name')
            addr_idx = header.index('business_address')
            cntry_idx = header.index('country')
            
            for line in f:
                parts = line.rstrip('\r\n').split('\t')
                if len(parts) <= max(id_idx, name_idx, addr_idx, cntry_idx):
                    continue
                cntry = parts[cntry_idx].strip()
                if cntry.lower() != target_lower:
                    continue
                
                eid = parts[id_idx].strip()
                name_cl = clean_text(parts[name_idx])
                addr_cl = clean_text(parts[addr_idx])
                cntry_cl = clean_text(cntry)
                
                data[eid] = (name_cl, addr_cl, cntry_cl)
                
                # Inverted index on non-stopword tokens
                tokens = set(name_cl.split()) - STOP_WORDS
                for tok in tokens:
                    if len(tok) >= 3:
                        token_index[tok].append(eid)
    
    # Prune ultra-frequent tokens (> 5000 occurrences) to avoid combinatorial explosion
    pruned_index = {tok: eids for tok, eids in token_index.items() if len(eids) <= 5000}
    return data, pruned_index


def process_country_s1(test_dir, country_target, s2s3_data, token_index, model, max_cands=15):
    """Stream S1 entities for country, generate candidates, extract features and predict."""
    target_lower = country_target.strip().lower()
    results = {}  # s1_id -> (candidate_ids_set, matched_ids_set)
    threshold = model.threshold
    
    # Collect matching S1 rows
    s1_rows = []
    with open(test_dir / 'test_source1.tsv', 'r', encoding='utf-8') as f:
        header = f.readline().rstrip('\r\n').split('\t')
        id_idx = header.index('entity_id')
        name_idx = header.index('business_name')
        addr_idx = header.index('business_address')
        cntry_idx = header.index('country')
        
        for line in f:
            parts = line.rstrip('\r\n').split('\t')
            if len(parts) <= max(id_idx, name_idx, addr_idx, cntry_idx):
                continue
            cntry = parts[cntry_idx].strip()
            if cntry.lower() == target_lower:
                s1_rows.append((
                    parts[id_idx].strip(),
                    clean_text(parts[name_idx]),
                    clean_text(parts[addr_idx]),
                    clean_text(cntry)
                ))
    
    print(f"  Processing {len(s1_rows):,} S1 entities for {country_target}...")
    
    BATCH_SIZE = 10000
    for b_start in tqdm(range(0, len(s1_rows), BATCH_SIZE), desc=f"Predicting {country_target}"):
        batch = s1_rows[b_start:b_start + BATCH_SIZE]
        
        batch_pairs = []
        batch_X = []
        batch_s1_cands = defaultdict(set)
        
        for s1_id, s1_name, s1_addr, s1_cntry in batch:
            tokens = set(s1_name.split()) - STOP_WORDS
            cand_counter = Counter()
            for tok in tokens:
                if len(tok) >= 3 and tok in token_index:
                    for cid in token_index[tok]:
                        cand_counter[cid] += 1
            
            # Select top candidate IDs
            if cand_counter:
                top_cands = [cid for cid, _ in cand_counter.most_common(max_cands)]
                cand_set = set(top_cands)
                batch_s1_cands[s1_id] = cand_set
                
                for cid in top_cands:
                    s2_name, s2_addr, s2_cntry = s2s3_data[cid]
                    feat = compute_features(s1_name, s1_addr, s1_cntry, s2_name, s2_addr, s2_cntry)
                    batch_X.append(feat)
                    batch_pairs.append((s1_id, cid))
            else:
                batch_s1_cands[s1_id] = set()
        
        # Run model inference on batch
        matched_dict = defaultdict(set)
        if batch_X:
            X_arr = np.array(batch_X, dtype=np.float32)
            probs = model.predict(X_arr)
            for i, prob in enumerate(probs):
                if prob >= threshold:
                    s1_id, cid = batch_pairs[i]
                    matched_dict[s1_id].add(cid)
        
        # Save results for this batch
        for s1_id, _, _, _ in batch:
            cands = batch_s1_cands.get(s1_id, set())
            matches = matched_dict.get(s1_id, set())
            # Integrity guarantee: matches must be subset of candidates
            matches = matches & cands
            results[s1_id] = (cands, matches)
            
    return results


def main():
    args = parse_args()
    test_dir = Path(args.test_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = args.model_path or str(output_dir / "model.joblib")
    
    print("=" * 50)
    print("PREDICTION PIPELINE")
    print("=" * 50)
    
    # 1. Load trained model
    print("\n[1] Loading trained model...")
    model = MatcherModel(random_seed=RANDOM_SEED)
    model.load(model_path)
    print(f"  Model threshold: {model.threshold:.2f}")
    
    # 2. Get list of all S1 entity IDs from test_source1.tsv
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
    
    # 3. Process country by country to keep memory bounded
    countries = ['France', 'US', 'India']
    all_results = {}
    
    for country in countries:
        print(f"\n[3] Processing Country: {country}...")
        s2s3_data, token_index = load_country_s2s3(test_dir, country)
        print(f"  Loaded {len(s2s3_data):,} S2/S3 entities and {len(token_index):,} blocking tokens for {country}")
        
        country_results = process_country_s1(
            test_dir, country, s2s3_data, token_index, model, max_cands=args.max_candidates
        )
        all_results.update(country_results)
        
        # Clean up memory
        del s2s3_data, token_index, country_results
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
    
    print(f"\nResults successfully written!")
    print(f"  Matching file: {matching_path}")
    print(f"  Candidate file: {candidate_path}")
    print(f"  Total S1 entities processed: {len(all_s1_ids):,}")
    print(f"  Total candidate pairs: {total_candidates:,} (avg {total_candidates/len(all_s1_ids):.1f}/entity)")
    print(f"  Total matched pairs: {total_matches:,} (avg {total_matches/len(all_s1_ids):.1f}/entity)")
    print(f"  Singletons (no match predicted): {singletons:,} ({singletons/len(all_s1_ids)*100:.1f}%)")


if __name__ == "__main__":
    main()
