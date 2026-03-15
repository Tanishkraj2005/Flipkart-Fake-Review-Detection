"""
Run this AFTER main.py to create a lightweight sample CSV for Streamlit Cloud.
Streamlit Cloud has a free 1GB memory limit, and the full 363K row file is too large.
This creates a 50K row stratified sample that keeps the fake/genuine ratio intact.
"""

import pandas as pd
import os

BASE = os.path.dirname(os.path.abspath(__file__))
INPUT  = os.path.join(BASE, "Data", "processed_reviews.csv")
OUTPUT = os.path.join(BASE, "Data", "processed_reviews.csv")

print("Loading processed dataset...")
df = pd.read_csv(INPUT, low_memory=False)
print(f"Loaded {len(df):,} rows")

# Stratified sample — keeps the same fake/genuine ratio
sample = (
    df.groupby("fake_status", group_keys=False)
    .apply(lambda x: x.sample(min(len(x), 25_000), random_state=42))
    .reset_index(drop=True)
)

sample.to_csv(OUTPUT, index=False)
print(f"Saved {len(sample):,} row sample → {OUTPUT}")
print("Now git add Data/processed_reviews.csv and push to GitHub.")
