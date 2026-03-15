import pandas as pd


def flag_exact_duplicates(df: pd.DataFrame, review_col: str = "Review") -> pd.DataFrame:
    normalised = df[review_col].str.lower().str.strip()
    counts = normalised.value_counts()

    df["exact_duplicate"] = normalised.map(counts) > 1
    return df


def duplicate_summary(df: pd.DataFrame) -> dict:
    exact_count = int(df.get("exact_duplicate", pd.Series(False)).sum())
    total = len(df)

    percentage = round((exact_count / total) * 100, 2) if total > 0 else 0

    return {
        "total_reviews": total,
        "exact_duplicates": exact_count,
        "duplicate_percentage": percentage
    }