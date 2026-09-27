import numpy as np
import math
from collections import Counter
from rapidfuzz import fuzz
from rapidfuzz.distance import Levenshtein


def get_token_overlap(name1, name2):
    tokens1 = set(name1.split()) if name1 else set()
    tokens2 = set(name2.split()) if name2 else set()
    if not tokens1 or not tokens2:
        return 0.0
    intersection = tokens1 & tokens2
    union = tokens1 | tokens2
    return len(intersection) / len(union)  # jaccard


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


def char_ngram_overlap(s1, s2, n=3):
    """Character n-gram Jaccard overlap — robust to word boundary corruption."""
    if len(s1) < n or len(s2) < n:
        return 0.0
    set1 = set(s1[i:i+n] for i in range(len(s1) - n + 1))
    set2 = set(s2[i:i+n] for i in range(len(s2) - n + 1))
    if not set1 or not set2:
        return 0.0
    return len(set1 & set2) / len(set1 | set2)


def sorted_token_containment(s1, s2):
    """How much of s1's tokens are contained in s2 (directional)."""
    t1 = set(s1.split()) if s1 else set()
    t2 = set(s2.split()) if s2 else set()
    if not t1:
        return 0.0
    return len(t1 & t2) / len(t1)


def nospace_ratio(s1, s2):
    """Similarity after stripping all spaces — catches concatenated corruptions."""
    a = s1.replace(' ', '')
    b = s2.replace(' ', '')
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return fuzz.ratio(a, b) / 100.0


def prefix_match_len(s1, s2, max_len=10):
    """Length of common prefix, normalized by max_len."""
    if not s1 or not s2:
        return 0.0
    common = 0
    for c1, c2 in zip(s1, s2):
        if c1 == c2:
            common += 1
        else:
            break
    return min(common, max_len) / max_len


def digit_overlap(addr1, addr2):
    """Jaccard overlap of numeric tokens in addresses."""
    import re
    nums1 = set(re.findall(r'\b\d+\b', addr1))
    nums2 = set(re.findall(r'\b\d+\b', addr2))
    if not nums1 or not nums2:
        return 0.0
    return len(nums1 & nums2) / len(nums1 | nums2)


def compute_features(name1, addr1, country1, name2, addr2, country2):
    """Compute 24 similarity features for a pair of entities."""
    features = []

    # --- NAME features (12) ---
    # rapidfuzz string similarity
    features.append(fuzz.ratio(name1, name2) / 100.0)
    features.append(fuzz.partial_ratio(name1, name2) / 100.0)
    features.append(fuzz.token_sort_ratio(name1, name2) / 100.0)
    features.append(fuzz.token_set_ratio(name1, name2) / 100.0)

    # token overlap (jaccard)
    features.append(get_token_overlap(name1, name2))

    # levenshtein (normalized)
    features.append(normalized_levenshtein(name1, name2))

    # TF-IDF cosine similarity
    features.append(tfidf_cosine_sim(name1, name2))

    # Character 3-gram overlap (catches corrupted spacing / concatenation)
    features.append(char_ngram_overlap(name1, name2, n=3))

    # No-space ratio (handles .com suffix and space-stripped names)
    features.append(nospace_ratio(name1, name2))

    # Prefix match length
    features.append(prefix_match_len(name1, name2))

    # Containment (directional)
    features.append(sorted_token_containment(name1, name2))

    # Name length ratio
    len1 = max(len(name1), 1)
    len2 = max(len(name2), 1)
    features.append(min(len1, len2) / max(len1, len2))

    # --- ADDRESS features (8) ---
    features.append(fuzz.ratio(addr1, addr2) / 100.0)
    features.append(fuzz.partial_ratio(addr1, addr2) / 100.0)
    features.append(fuzz.token_sort_ratio(addr1, addr2) / 100.0)

    # address token overlap (jaccard)
    features.append(get_token_overlap(addr1, addr2))

    # address levenshtein (normalized)
    features.append(normalized_levenshtein(addr1, addr2))

    # address TF-IDF cosine similarity
    features.append(tfidf_cosine_sim(addr1, addr2))

    # address digit overlap
    features.append(digit_overlap(addr1, addr2))

    # address length ratio
    alen1 = max(len(addr1), 1)
    alen2 = max(len(addr2), 1)
    features.append(min(alen1, alen2) / max(alen1, alen2))

    # --- CROSS features (4) ---
    # country match (1 or 0)
    features.append(1.0 if country1 == country2 else 0.0)

    # name-in-address cross signal
    features.append(fuzz.partial_ratio(name1, addr2) / 100.0)

    # address-in-name cross signal
    features.append(fuzz.partial_ratio(addr1, name2) / 100.0)

    # combined name+addr similarity
    combined1 = (name1 + ' ' + addr1).strip()
    combined2 = (name2 + ' ' + addr2).strip()
    features.append(fuzz.token_sort_ratio(combined1, combined2) / 100.0)

    return np.array(features, dtype=np.float32)


FEATURE_NAMES = [
    'name_ratio', 'name_partial_ratio', 'name_token_sort', 'name_token_set',
    'name_jaccard', 'name_levenshtein', 'name_tfidf_cosine',
    'name_char3gram', 'name_nospace_ratio', 'name_prefix_match',
    'name_containment', 'name_len_ratio',
    'addr_ratio', 'addr_partial_ratio', 'addr_token_sort',
    'addr_jaccard', 'addr_levenshtein', 'addr_tfidf_cosine',
    'addr_digit_overlap', 'addr_len_ratio',
    'country_match', 'name_in_addr', 'addr_in_name', 'combined_token_sort'
]
