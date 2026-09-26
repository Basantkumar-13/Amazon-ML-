import pandas as pd
import regex as re
from pathlib import Path


def clean_text(text):
    if pd.isna(text):
        return ""
    text = str(text).lower().strip()
    # remove extra spaces
    text = re.sub(r'\s+', ' ', text)
    # remove special chars but keep letters, numbers, spaces
    text = re.sub(r'[^\w\s]', '', text)
    return text


def load_source(filepath):
    df = pd.read_csv(filepath, sep='\t', dtype=str)
    df = df.fillna("")
    # clean business name and address
    df['name_clean'] = df['business_name'].apply(clean_text)
    df['addr_clean'] = df['business_address'].apply(clean_text)
    df['country_clean'] = df['country'].apply(clean_text)
    return df


def load_ground_truth(filepath):
    df = pd.read_csv(filepath, sep='\t', dtype=str)
    df['matched_entity_ids'] = df['matched_entity_ids'].fillna("")
    # handle singletons (empty matched_entity_ids) properly
    df['match_list'] = df['matched_entity_ids'].apply(
        lambda x: [m.strip() for m in x.split(',') if m.strip()] if x.strip() else []
    )
    return df


def load_all_train_data(train_dir):
    train_dir = Path(train_dir)
    s1 = load_source(train_dir / "train_source1.tsv")
    s2 = load_source(train_dir / "train_source2.tsv")
    s3 = load_source(train_dir / "train_source3.tsv")
    gt = load_ground_truth(train_dir / "train_ground_truth.tsv")
    return s1, s2, s3, gt


def load_all_test_data(test_dir):
    test_dir = Path(test_dir)
    s1 = load_source(test_dir / "test_source1.tsv")
    s2 = load_source(test_dir / "test_source2.tsv")
    s3 = load_source(test_dir / "test_source3.tsv")
    return s1, s2, s3
