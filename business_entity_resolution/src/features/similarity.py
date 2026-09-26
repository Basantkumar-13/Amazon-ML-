import numpy as np
from rapidfuzz import fuzz
from rapidfuzz.distance import Levenshtein
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def get_token_overlap(name1, name2):
    tokens1 = set(name1.split()) if name1 else set()
    tokens2 = set(name2.split()) if name2 else set()
    if not tokens1 or not tokens2:
        return 0.0
    intersection = tokens1 & tokens2
    union = tokens1 | tokens2
    return len(intersection) / len(union)  # jaccard


import math
from collections import Counter


def tfidf_cosine_sim(text1, text2):
    tokens1 = text1.split()
    tokens2 = text2.split()
    if not tokens1 or not tokens2:
        return 0.0
    c1 = Counter(tokens1)
    c2 = Counter(tokens2)
    # Exact smooth IDF for 2 documents (matching sklearn default):
    # idf = ln((1 + 2) / (1 + df)) + 1
    # df=2 -> idf=1.0; df=1 -> idf=1.4054651081081644
    IDF_SINGLE = 1.4054651081081644
    dot = 0.0
    norm1 = 0.0
    norm2 = 0.0
    for w, tf1 in c1.items():
        v1 = tf1 * (1.0 if w in c2 else IDF_SINGLE)
        norm1 += v1 * v1
        if w in c2:
            dot += v1 * (c2[w] * 1.0)
    for w, tf2 in c2.items():
        v2 = tf2 * (1.0 if w in c1 else IDF_SINGLE)
        norm2 += v2 * v2
    if norm1 == 0.0 or norm2 == 0.0:
        return 0.0
    return float(dot / (math.sqrt(norm1) * math.sqrt(norm2)))


def normalized_levenshtein(s1, s2):
    if not s1 and not s2:
        return 1.0
    if not s1 or not s2:
        return 0.0
    dist = Levenshtein.distance(s1, s2)
    max_len = max(len(s1), len(s2))
    return 1.0 - (dist / max_len)


def compute_features(name1, addr1, country1, name2, addr2, country2):
    features = []

    # name similarity features (rapidfuzz)
    features.append(fuzz.ratio(name1, name2) / 100.0)
    features.append(fuzz.partial_ratio(name1, name2) / 100.0)
    features.append(fuzz.token_sort_ratio(name1, name2) / 100.0)
    features.append(fuzz.token_set_ratio(name1, name2) / 100.0)

    # name token overlap (jaccard)
    features.append(get_token_overlap(name1, name2))

    # name levenshtein (normalized)
    features.append(normalized_levenshtein(name1, name2))

    # name TF-IDF cosine similarity
    features.append(tfidf_cosine_sim(name1, name2))

    # address similarity features
    features.append(fuzz.ratio(addr1, addr2) / 100.0)
    features.append(fuzz.partial_ratio(addr1, addr2) / 100.0)
    features.append(fuzz.token_sort_ratio(addr1, addr2) / 100.0)

    # address token overlap (jaccard)
    features.append(get_token_overlap(addr1, addr2))

    # address levenshtein (normalized)
    features.append(normalized_levenshtein(addr1, addr2))

    # address TF-IDF cosine similarity
    features.append(tfidf_cosine_sim(addr1, addr2))

    # country match (1 or 0)
    features.append(1.0 if country1 == country2 else 0.0)

    # name length ratio
    len1 = max(len(name1), 1)
    len2 = max(len(name2), 1)
    features.append(min(len1, len2) / max(len1, len2))

    # address length ratio
    alen1 = max(len(addr1), 1)
    alen2 = max(len(addr2), 1)
    features.append(min(alen1, alen2) / max(alen1, alen2))

    return np.array(features, dtype=np.float32)


FEATURE_NAMES = [
    'name_ratio', 'name_partial_ratio', 'name_token_sort', 'name_token_set',
    'name_jaccard', 'name_levenshtein', 'name_tfidf_cosine',
    'addr_ratio', 'addr_partial_ratio', 'addr_token_sort',
    'addr_jaccard', 'addr_levenshtein', 'addr_tfidf_cosine',
    'country_match', 'name_len_ratio', 'addr_len_ratio'
]
