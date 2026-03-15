import pandas as pd
import re
from src.clean_text import caps_ratio
from src.flags import (
    length_flag, repetition_flag, generic_flag,
    rating_mismatch_flag, meaningless_flag,
    duplicate_flag, caps_ratio_flag, short_review_flag,
)

from src.sentiment import sentiment_score

def add_basic_features(df: pd.DataFrame) -> pd.DataFrame:
                                                      
    df["review_length"] = df["Summary"].apply(len)
    df["word_count"] = df["Summary"].apply(lambda x: len(x.split()))
    df["caps_ratio"] = df["Summary"].apply(caps_ratio)
    df["sentence_count"] = df["Summary"].apply(
        lambda x: max(1, len(re.split(r"[.!?]+", x)))
    )


    df["avg_word_length"] = df["Summary"].apply(
        lambda x: (
            sum(len(w) for w in x.split()) / len(x.split())
            if x.split() else 0
        )
    )
    df["exclamation_count"] = df["Summary"].apply(lambda x: x.count("!"))

    return df


def add_sentiment(df: pd.DataFrame) -> pd.DataFrame:
                                             
    print("→ Computing sentiment scores (this may take a moment)...")

    df["sentiment_score"] = df["Summary"].apply(sentiment_score)
    return df

def add_duplicate_flag(df: pd.DataFrame) -> pd.DataFrame:
                                                                     
    review_counts = df["Summary"].str.lower().str.strip().value_counts().to_dict()
    df["duplicate_flag"] = df["Summary"].apply(
        lambda x: duplicate_flag(x, review_counts)
    )

    return df


def add_rule_flags(df: pd.DataFrame) -> pd.DataFrame:
                                          
    df["length_flag"]          = df["Summary"].apply(length_flag)
    df["repetition_flag"]      = df["Summary"].apply(repetition_flag)
    df["generic_flag"]         = df["Summary"].apply(generic_flag)
    df["meaningless_flag"]     = df["Summary"].apply(meaningless_flag)
    df["caps_ratio_flag"]      = df["Summary"].apply(caps_ratio_flag)
    df["short_review_flag"]    = df["Summary"].apply(short_review_flag)
    df["rating_mismatch_flag"] = df.apply(
        lambda row: rating_mismatch_flag(row["Summary"], row["Rate"]),
        axis=1,
    )
    return df

def add_fraud_score(df: pd.DataFrame) -> pd.DataFrame:

    df["fraud_score"] = (
        df["length_flag"]           * 1
        + df["repetition_flag"]     * 1
        + df["generic_flag"]        * 1
        + df["rating_mismatch_flag"]* 1
        + df["duplicate_flag"]      * 1
        + df["meaningless_flag"]    * 1
        + df["caps_ratio_flag"]     * 1
        + df["short_review_flag"]   * 1
    )
    return df

def build_features(df: pd.DataFrame) -> pd.DataFrame:

    print("[Feature Engineering] Adding basic text features...")
    df = add_basic_features(df)
    print("[Feature Engineering] Adding duplicate flags...")
    df = add_duplicate_flag(df)
    print("[Feature Engineering] Adding rule-based flags...")
    df = add_rule_flags(df)
    print("[Feature Engineering] Adding sentiment scores...")
    df = add_sentiment(df)
    print("[Feature Engineering] Computing fraud score...")
    df = add_fraud_score(df)

    return df
