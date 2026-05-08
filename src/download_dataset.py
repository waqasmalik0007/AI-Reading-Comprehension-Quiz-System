"""
download_dataset.py — Download and prepare the RACE dataset from Kaggle.

Dataset: https://www.kaggle.com/datasets/ankitdhiman7/race-dataset

Usage:
    python src/download_dataset.py

Requirements:
    - kaggle package installed (pip install kaggle)
    - Kaggle API key configured at ~/.kaggle/kaggle.json
      (download from https://www.kaggle.com/settings -> API -> Create New Token)
"""

import os
import sys
import zipfile
import shutil
import pandas as pd
from sklearn.model_selection import train_test_split

DATA_RAW_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'raw')
os.makedirs(DATA_RAW_DIR, exist_ok=True)

KAGGLE_DATASET = 'ankitdhiman7/race-dataset'
EXPECTED_COLUMNS = ['id', 'article', 'question', 'A', 'B', 'C', 'D', 'answer']


def check_kaggle_credentials():
    """Check if Kaggle API credentials are configured."""
    kaggle_json = os.path.expanduser('~/.kaggle/kaggle.json')
    win_path = os.path.join(os.path.expanduser('~'), '.kaggle', 'kaggle.json')

    if os.path.exists(kaggle_json) or os.path.exists(win_path):
        print("[OK] Kaggle credentials found.")
        return True

    print("[ERROR] Kaggle credentials not found!")
    print("\nTo set up Kaggle API:")
    print("  1. Go to https://www.kaggle.com/settings")
    print("  2. Scroll to 'API' section -> click 'Create New Token'")
    print("  3. It downloads 'kaggle.json'")
    print(f"  4. Move it to: {win_path}")
    print("  5. Run this script again")
    return False


def download_from_kaggle():
    """Download RACE dataset using Kaggle API."""
    if not check_kaggle_credentials():
        return False

    try:
        import kaggle
        print(f"\nDownloading dataset: {KAGGLE_DATASET}")
        kaggle.api.authenticate()
        kaggle.api.dataset_download_files(
            KAGGLE_DATASET,
            path=DATA_RAW_DIR,
            unzip=True
        )
        print(f"[OK] Downloaded to: {DATA_RAW_DIR}")
        return True
    except Exception as e:
        print(f"[ERROR] Download failed: {e}")
        return False


def find_csv_file():
    """Find the main CSV file in data/raw/."""
    for fname in os.listdir(DATA_RAW_DIR):
        if fname.endswith('.csv') and 'race' in fname.lower():
            return os.path.join(DATA_RAW_DIR, fname)

    # Try any CSV
    csvs = [f for f in os.listdir(DATA_RAW_DIR) if f.endswith('.csv')]
    if csvs:
        return os.path.join(DATA_RAW_DIR, csvs[0])

    return None


def normalize_dataframe(df):
    """Normalize column names to expected format."""
    df.columns = [c.strip().lower() for c in df.columns]

    rename_map = {}
    for col in df.columns:
        if 'article' in col or 'passage' in col or 'context' in col:
            rename_map[col] = 'article'
        elif 'question' in col:
            rename_map[col] = 'question'
        elif col in ['option_a', 'option1', 'opt_a', 'a']:
            rename_map[col] = 'A'
        elif col in ['option_b', 'option2', 'opt_b', 'b']:
            rename_map[col] = 'B'
        elif col in ['option_c', 'option3', 'opt_c', 'c']:
            rename_map[col] = 'C'
        elif col in ['option_d', 'option4', 'opt_d', 'd']:
            rename_map[col] = 'D'
        elif 'answer' in col or 'label' in col:
            rename_map[col] = 'answer'

    df = df.rename(columns=rename_map)

    if 'A' not in df.columns:
        df.columns = [c.upper() if len(c) == 1 else c for c in df.columns]

    if 'id' not in df.columns:
        df['id'] = [f'race_{i}' for i in range(len(df))]

    return df


def split_and_save(df):
    """Split into train/val/test and save as CSVs."""
    print(f"\nDataset shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print(f"Answer distribution:\n{df['answer'].value_counts()}")

    # 80% train, 10% val, 10% test
    train_df, temp_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['answer'])
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42, stratify=temp_df['answer'])

    train_df.to_csv(os.path.join(DATA_RAW_DIR, 'train.csv'), index=False)
    val_df.to_csv(os.path.join(DATA_RAW_DIR, 'val.csv'), index=False)
    test_df.to_csv(os.path.join(DATA_RAW_DIR, 'test.csv'), index=False)

    print(f"\n[OK] Splits saved:")
    print(f"  Train: {len(train_df)} rows -> data/raw/train.csv")
    print(f"  Val:   {len(val_df)} rows -> data/raw/val.csv")
    print(f"  Test:  {len(test_df)} rows -> data/raw/test.csv")


def main():
    print("=" * 60)
    print("  RACE Dataset Download & Preparation")
    print("  Source: Kaggle — ankitdhiman7/race-dataset")
    print("=" * 60)

    # Check if already downloaded
    train_path = os.path.join(DATA_RAW_DIR, 'train.csv')
    if os.path.exists(train_path):
        df = pd.read_csv(train_path)
        print(f"[INFO] train.csv already exists ({len(df)} rows). Skipping download.")
        print("       Delete data/raw/train.csv to force re-download.")
        return

    # Try Kaggle download
    success = download_from_kaggle()

    if not success:
        print("\n[MANUAL FALLBACK] Download the CSV manually:")
        print(f"  1. Visit: https://www.kaggle.com/datasets/{KAGGLE_DATASET}")
        print("  2. Click 'Download'")
        print(f"  3. Extract and place the CSV in: {DATA_RAW_DIR}")
        print("  4. Run this script again")
        sys.exit(1)

    # Find and load CSV
    csv_path = find_csv_file()
    if csv_path is None:
        print("[ERROR] No CSV file found after download.")
        sys.exit(1)

    print(f"\nLoading: {csv_path}")
    df = pd.read_csv(csv_path)
    df = normalize_dataframe(df)

    # Validate
    missing = [c for c in ['article', 'question', 'A', 'B', 'C', 'D', 'answer'] if c not in df.columns]
    if missing:
        print(f"[WARN] Missing columns: {missing}")
        print(f"Available columns: {list(df.columns)}")

    df = df.dropna(subset=['article', 'question', 'answer'])
    split_and_save(df)
    print("\n[DONE] Dataset ready. Run preprocessing next:")
    print("       python src/preprocessing.py")


if __name__ == '__main__':
    main()
