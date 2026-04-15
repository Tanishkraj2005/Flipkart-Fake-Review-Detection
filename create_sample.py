import pandas as pd
import os

BASE = os.path.dirname(os.path.abspath(__file__))
INPUT  = os.path.join(BASE, "Data", "processed_reviews.csv")
OUTPUT = os.path.join(BASE, "Data", "processed_reviews.csv")

print("Loading processed dataset...")
df = pd.read_csv(INPUT, low_memory=False)
print(f"Loaded {len(df):,} rows")

sample = (
    df.groupby("fake_status", group_keys=False)
    .apply(lambda x: x.sample(min(len(x), 25_000), random_state=42))
    .reset_index(drop=True)
)

sample.to_csv(OUTPUT, index=False)
print(f"Saved {len(sample):,} row sample → {OUTPUT}")
