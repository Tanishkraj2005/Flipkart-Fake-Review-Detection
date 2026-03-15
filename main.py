import os
import time
import sys
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()

sys.path.insert(0, BASE_DIR)

from src.clean_text import clean_text, clean_product_name
from src.feature_engineering import build_features
from src.fake_detector import detect_fake
from src.duplicate_detection import flag_exact_duplicates
from src.reviewer_detection import build_reviewer_profile
from src.sentiment import sentiment_label
from src.ml_model import run_ml_model


INPUT_CSV = os.path.join(BASE_DIR, "Data", "flipkart_reviews.csv")
PROCESSED_CSV = os.path.join(BASE_DIR, "Data", "processed_reviews.csv")
POWERBI_CSV = os.path.join(BASE_DIR, "dashboard_data", "powerbi_dataset.csv")

os.makedirs(os.path.join(BASE_DIR, "Data"), exist_ok=True)
os.makedirs(os.path.join(BASE_DIR, "dashboard_data"), exist_ok=True)


def load_data(path: str) -> pd.DataFrame:

    print("\n============================================================")
    print("AI-Powered Flipkart Review Fraud Analytics System")
    print("============================================================\n")

    print("[1/9] Loading dataset...")

    df = pd.read_csv(
        path,
        encoding="latin1",
        on_bad_lines="skip",
        low_memory=False,
    )

    print(f"✓ Loaded {len(df):,} reviews | Columns: {list(df.columns)}")

    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:

    print("\n[2/9] Cleaning dataset...")

    df["Rate"] = pd.to_numeric(df["Rate"], errors="coerce")
    df["Rate"] = df["Rate"].fillna(df["Rate"].median())

    if "Price" in df.columns:

        df["Price"] = (
            df["Price"]
            .astype(str)
            .str.replace("?", "", regex=False)
            .str.replace(",", "", regex=False)
            .str.strip()
        )

        df["Price"] = pd.to_numeric(df["Price"], errors="coerce")

    df["Product_name"] = df["Product_name"].apply(clean_product_name)
    df["Review"] = df["Review"].apply(clean_text)
    df["Summary"] = df["Summary"].apply(clean_text)

    original_len = len(df)

    mask = (
        df["Review"].str.strip().str.len() > 0
    ) & (
        df["Summary"].str.strip().str.len() > 0
    )

    df = df.loc[mask].reset_index(drop=True)

    print(f"✓ Dropped {original_len - len(df):,} rows with empty reviews/summaries")
    print(f"✓ Clean dataset: {len(df):,} reviews")

    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:

    print("\n[3/9] Engineering features...")

    df = build_features(df)

    print(
        "✓ Features added: review_length, word_count, caps_ratio, "
        "sentiment_score, and fraud flags"
    )

    return df


def run_detection(df: pd.DataFrame) -> pd.DataFrame:

    print("\n[4/9] Running rule-based fake review detection...")

    df["fake_status"] = df.apply(detect_fake, axis=1)

    counts = df["fake_status"].value_counts()

    print(f"✓ Genuine:      {counts.get('Genuine', 0):,}")
    print(f"✓ Likely Fake:  {counts.get('Likely Fake', 0):,}")

    return df


def run_duplicate_detection(df: pd.DataFrame) -> pd.DataFrame:

    print("\n[6/9] Running duplicate detection...")

    df = flag_exact_duplicates(df, review_col="Summary")

    print(f"✓ Exact duplicates: {df['exact_duplicate'].sum():,}")

    return df


def run_reviewer_scoring(df: pd.DataFrame) -> pd.DataFrame:

    print("\n[7/9] Computing sentiment labels...")

    df["sentiment_label"] = df["sentiment_score"].apply(sentiment_label)

    return df


def export_processed(df: pd.DataFrame) -> None:

    print("\n[8/9] Saving processed dataset...")

    output_cols = [
        "Product_name",
        "Price",
        "Rate",
        "Review",
        "Summary",
        "review_length",
        "word_count",
        "caps_ratio",
        "avg_word_length",
        "sentiment_score",
        "sentiment_label",
        "length_flag",
        "repetition_flag",
        "generic_flag",
        "rating_mismatch_flag",
        "duplicate_flag",
        "meaningless_flag",
        "caps_ratio_flag",
        "exact_duplicate",
        "fraud_score",
        "fake_status",
        "ml_prediction",
    ]

    output_cols = [c for c in output_cols if c in df.columns]

    df[output_cols].to_csv(PROCESSED_CSV, index=False)

    print(f"✓ Saved → {PROCESSED_CSV}")


def export_powerbi(df: pd.DataFrame) -> None:

    print("\n[9/9] Exporting Power BI dataset...")

    powerbi_cols = [
        "Product_name",
        "Review",
        "Summary",
        "Price",
        "Rate",
        "review_length",
        "word_count",
        "sentiment_score",
        "sentiment_label",
        "fraud_score",
        "fake_status",
        "exact_duplicate",
        "ml_prediction",
    ]

    powerbi_cols = [c for c in powerbi_cols if c in df.columns]

    df[powerbi_cols].to_csv(POWERBI_CSV, index=False)

    print(f"✓ Saved → {POWERBI_CSV}")





def main():

    t0 = time.time()

    df = load_data(INPUT_CSV)

    df = clean_data(df)

    df = engineer_features(df)

    df = run_detection(df)

    print("\n[5/9] Training Machine Learning Model...")

    df = run_ml_model(df)

    df = run_duplicate_detection(df)

    df = run_reviewer_scoring(df)

    export_processed(df)

    export_powerbi(df)


    elapsed = time.time() - t0

    print(f"Total runtime: {elapsed:.1f} seconds\n")


if __name__ == "__main__":

    main()