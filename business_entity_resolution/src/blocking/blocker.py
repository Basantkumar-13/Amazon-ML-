import pandas as pd
from rapidfuzz import fuzz
from tqdm import tqdm


def get_name_tokens(name):
    if not name:
        return set()
    return set(name.split())


def block_by_country_and_tokens(s1_df, s2s3_df, min_token_overlap=1):
    """
    Blocking step: only compare entities that share same country
    AND have at least 1 common word in their business name.
    This massively reduces the number of pairs we need to compare.
    """
    # build an inverted index: for each(country, token)-> list of s2/s3 entity indices
    token_index = {}
    for idx, row in s2s3_df.iterrows():
        country = row['country_clean']
        tokens = get_name_tokens(row['name_clean'])
        for tok in tokens:
            if len(tok) >= 2:  # skip single char tokens
                key = (country, tok)
                if key not in token_index:
                    token_index[key] = []
                token_index[key].append(idx)

    # for each s1 entity, find candidate s2/s3 entities
    candidate_pairs = []
    for idx, row in tqdm(s1_df.iterrows(), total=len(s1_df), desc="Blocking"):
        country = row['country_clean']
        tokens = get_name_tokens(row['name_clean'])
        
        # collect all candidate indices from the inverted index
        candidate_indices = set()
        for tok in tokens:
            if len(tok) >= 2:
                key = (country, tok)
                if key in token_index:
                    for c_idx in token_index[key]:
                        candidate_indices.add(c_idx)
        
        s1_id = row['entity_id']
        for c_idx in candidate_indices:
            candidate_pairs.append((s1_id, s2s3_df.at[c_idx, 'entity_id']))

    return candidate_pairs
