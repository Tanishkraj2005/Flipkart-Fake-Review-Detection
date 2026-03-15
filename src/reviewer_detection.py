import pandas as pd

def build_reviewer_profile(df: pd.DataFrame) -> pd.DataFrame:

    df_fake = df.copy()
    df_fake["is_fake"] = df_fake["fake_status"].isin(

        ["Suspicious", "Likely Fake"]

    ).astype(int)

    profile = (
        df_fake.groupby("Product_name")
        .agg(
            total_reviews=("Review", "count"),
            fake_review_count=("is_fake", "sum"),
            avg_fraud_score=("fraud_score", "mean"),
            avg_sentiment=("sentiment_score", "mean"),
            avg_rating=("Rate", "mean"),
        )

        .reset_index()

    )

    profile["fake_review_pct"] = (
        profile["fake_review_count"] / profile["total_reviews"] * 100
    ).round(2)

    profile = profile.sort_values("fake_review_pct", ascending=False)
    return profile
