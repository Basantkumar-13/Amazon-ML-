
import argparse
import os
import sys
from typing import Dict, List, Set


def parse_args():
    parser = argparse.ArgumentParser(description="Validate submission TSVs for Amazon ML Challenge 2026")
    parser.add_argument("--matching", required=True, help="Path to matching_results.tsv")
    parser.add_argument("--candidate", required=True, help="Path to candidate_pairs.tsv")
    parser.add_argument("--test-dir", required=True, help="Path to test dataset directory containing test_source1.tsv, etc.")
    return parser.parse_args()


def load_test_entity_ids(test_dir: str):
    s1_ids = set()
    s2_s3_ids = set()

    s1_path = os.path.join(test_dir, "test_source1.tsv")
    s2_path = os.path.join(test_dir, "test_source2.tsv")
    s3_path = os.path.join(test_dir, "test_source3.tsv")

    for path, target_set in [(s1_path, s1_ids), (s2_path, s2_s3_ids), (s3_path, s2_s3_ids)]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Required test file not found: {path}")
        with open(path, "r", encoding="utf-8") as f:
            header = f.readline().rstrip("\r\n").split("\t")
            try:
                id_col = header.index("entity_id")
            except ValueError:
                raise ValueError(f"Column 'entity_id' not found in {path}")
            for line in f:
                parts = line.rstrip("\r\n").split("\t")
                if parts and len(parts) > id_col:
                    target_set.add(parts[id_col].strip())

    return s1_ids, s2_s3_ids


def parse_submission_file(file_path: str, expected_header: tuple, id_col_name: str, issues: List[str]) -> Dict[str, List[str]]:
    if not os.path.exists(file_path):
        issues.append(f"File not found: {file_path}")
        return {}

    records = {}
    with open(file_path, "r", encoding="utf-8") as f:
        first_line = f.readline()
        if not first_line:
            issues.append(f"{file_path} is empty.")
            return {}

        header = first_line.rstrip("\r\n").split("\t")
        if tuple(header) != expected_header:
            issues.append(
                f"{file_path} invalid header: found {header}, expected {list(expected_header)}"
            )

        for line_num, line in enumerate(f, start=2):
            raw = line.rstrip("\r\n")
            if not raw:
                continue
            cols = raw.split("\t")
            if len(cols) == 1:
                # Means matched/candidate ids is empty string (singleton)
                s1_id = cols[0].strip()
                id_list = []
            elif len(cols) == 2:
                s1_id = cols[0].strip()
                ids_str = cols[1].strip()
                id_list = [x.strip() for x in ids_str.split(",") if x.strip()] if ids_str else []
            else:
                issues.append(f"{file_path} line {line_num}: expected 2 tab-separated columns, found {len(cols)}")
                continue

            if s1_id in records:
                issues.append(f"{file_path} line {line_num}: duplicate source1_entity_id '{s1_id}'")
            else:
                records[s1_id] = id_list

    return records


def validate(matching_path: str, candidate_path: str, test_dir: str):
    issues = []

    print("[1/4] Loading test reference IDs...")
    try:
        expected_s1_ids, valid_s2_s3_ids = load_test_entity_ids(test_dir)
        print(f"      Loaded {len(expected_s1_ids):,} Source 1 test IDs and {len(valid_s2_s3_ids):,} Source 2/3 test IDs.")
    except Exception as e:
        issues.append(f"Error loading test files: {e}")
        return issues

    print("[2/4] Parsing matching_results.tsv...")
    matching_records = parse_submission_file(
        matching_path,
        ("source1_entity_id", "matched_entity_ids"),
        "matched_entity_ids",
        issues
    )

    print("[3/4] Parsing candidate_pairs.tsv...")
    candidate_records = parse_submission_file(
        candidate_path,
        ("source1_entity_id", "candidate_entity_ids"),
        "candidate_entity_ids",
        issues
    )

    print("[4/4] Cross-checking rules and constraints...")
    # Rule 1: Every S1 entity in test must appear in matching_results
    missing_in_matching = expected_s1_ids - set(matching_records.keys())
    if missing_in_matching:
        sample = list(missing_in_matching)[:5]
        issues.append(f"matching_results.tsv missing {len(missing_in_matching)} Source 1 test entities (e.g. {sample})")

    extra_in_matching = set(matching_records.keys()) - expected_s1_ids
    if extra_in_matching:
        sample = list(extra_in_matching)[:5]
        issues.append(f"matching_results.tsv has {len(extra_in_matching)} unexpected entities not in test set (e.g. {sample})")

    # Rule 2: Every S1 entity in test must appear in candidate_pairs
    missing_in_candidate = expected_s1_ids - set(candidate_records.keys())
    if missing_in_candidate:
        sample = list(missing_in_candidate)[:5]
        issues.append(f"candidate_pairs.tsv missing {len(missing_in_candidate)} Source 1 test entities (e.g. {sample})")

    # Rule 3: Valid target IDs and duplicate checks
    for s1_id, match_ids in matching_records.items():
        if len(match_ids) != len(set(match_ids)):
            issues.append(f"matching_results.tsv for {s1_id}: duplicate IDs found in matched_entity_ids")
        for tid in match_ids:
            if not (tid.startswith("S2-") or tid.startswith("S3-")):
                issues.append(f"matching_results.tsv for {s1_id}: invalid ID prefix in '{tid}' (must be S2- or S3-)")
            if tid not in valid_s2_s3_ids:
                issues.append(f"matching_results.tsv for {s1_id}: entity '{tid}' does not exist in test set")

    for s1_id, cand_ids in candidate_records.items():
        if len(cand_ids) != len(set(cand_ids)):
            issues.append(f"candidate_pairs.tsv for {s1_id}: duplicate IDs found in candidate_entity_ids")
        for tid in cand_ids:
            if not (tid.startswith("S2-") or tid.startswith("S3-")):
                issues.append(f"candidate_pairs.tsv for {s1_id}: invalid ID prefix in '{tid}' (must be S2- or S3-)")
            if tid not in valid_s2_s3_ids:
                issues.append(f"candidate_pairs.tsv for {s1_id}: entity '{tid}' does not exist in test set")

    # Rule 4: Final matches must be a subset of candidate pairs
    cand_mismatch_count = 0
    for s1_id, match_ids in matching_records.items():
        cand_ids_set = set(candidate_records.get(s1_id, []))
        for m in match_ids:
            if m not in cand_ids_set:
                cand_mismatch_count += 1
                if cand_mismatch_count <= 5:
                    issues.append(f"Pipeline integrity warning: {m} is matched for {s1_id} but never appeared in candidate_pairs")

    if cand_mismatch_count > 5:
        issues.append(f"... and {cand_mismatch_count - 5} more matches not in candidate pairs.")

    return issues


def main():
    args = parse_args()
    issues = validate(args.matching, args.candidate, args.test_dir)

    if not issues:
        print(" PASS: Submission is valid and ready!")
        
        sys.exit(0)
    else:

        print(f" FAIL: Found {len(issues)} issue(s):")
       
        for i, issue in enumerate(issues[:25], start=1):
            print(f" {i}. {issue}")
        if len(issues) > 25:
            print(f" ... and {len(issues) - 25} additional issues omitted.")
        sys.exit(1)


if __name__ == "__main__":
    main()
