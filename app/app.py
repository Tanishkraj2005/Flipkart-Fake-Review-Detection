import os
import sys
import pandas as pd
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.clean_text import clean_text
from src.flags import (
    length_flag, repetition_flag, generic_flag,
    rating_mismatch_flag, meaningless_flag, caps_ratio_flag,
)
from src.sentiment import sentiment_score, sentiment_label
from src.fake_detector import fake_probability
from src.reviewer_detection import build_reviewer_profile

st.set_page_config(
    page_title="Flipkart Review Fraud Analytics",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
    .metric-card {
        background: #1e1e2e;
        border-radius: 12px;
        padding: 18px 22px;
        border-left: 4px solid #7c3aed;
        margin-bottom: 8px;
    }
    .metric-label { color: #94a3b8; font-size: 13px; font-weight: 600; letter-spacing: .5px; }
    .metric-value { color: #f1f5f9; font-size: 32px; font-weight: 800; line-height: 1.1; }
    .metric-sub   { color: #7c3aed; font-size: 13px; }
    .flag-row { display: flex; align-items: center; gap: 10px; padding: 6px 0; border-bottom: 1px solid #2d2d3f; }
    .flag-dot-red  { width:12px; height:12px; border-radius:50%; background:#ef4444; display:inline-block; }
    .flag-dot-green{ width:12px; height:12px; border-radius:50%; background:#22c55e; display:inline-block; }
</style>
""", unsafe_allow_html=True)

@st.cache_data(show_spinner="Loading dataset…")
def load_data() -> pd.DataFrame:
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path = os.path.join(base, "Data", "processed_reviews.csv")
    if not os.path.exists(path):
        return pd.DataFrame()
    df = pd.read_csv(path, nrows=100_000, low_memory=False)
    if "exact_duplicate" in df.columns:
        df["exact_duplicate"] = df["exact_duplicate"].astype(bool)
    return df

df = load_data()
has_data = not df.empty

st.markdown(
    "<h1 style='text-align:center; font-size:2.2rem;'>🛡️ Flipkart Review Fraud Analytics</h1>"
    "<p style='text-align:center; color:#94a3b8; margin-top:-8px;'>"
    "Rule-based fraud detection · Sentiment analysis · 363K reviews</p>",
    unsafe_allow_html=True,
)
st.divider()

tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Dashboard",
    "🔍 Review Analyzer",
    "🏭 Product Intelligence",
    "🗄️ Dataset Explorer",
])

with tab1:
    if not has_data:
        st.warning("⚠️ No processed data found. Run `python main.py` first to generate the dataset.")
        st.stop()

    total       = len(df)
    fake_count  = (df["fake_status"] == "Likely Fake").sum()
    genuine_ct  = (df["fake_status"] == "Genuine").sum()
    dup_count   = df["exact_duplicate"].sum() if "exact_duplicate" in df.columns else 0
    avg_fraud   = df["fraud_score"].mean() if "fraud_score" in df.columns else 0

    c1, c2, c3, c4 = st.columns(4)

    def kpi(col, label, value, sub, color="#7c3aed"):
        col.markdown(
            f"<div class='metric-card' style='border-color:{color}'>"
            f"<div class='metric-label'>{label}</div>"
            f"<div class='metric-value'>{value}</div>"
            f"<div class='metric-sub'>{sub}</div></div>",
            unsafe_allow_html=True,
        )

    kpi(c1, "TOTAL REVIEWS",   f"{total:,}",           "in this sample",          "#7c3aed")
    kpi(c2, "LIKELY FAKE",     f"{fake_count:,}",       f"{fake_count/total*100:.1f}% of dataset",  "#ef4444")
    kpi(c3, "GENUINE",         f"{genuine_ct:,}",       f"{genuine_ct/total*100:.1f}% of dataset",  "#22c55e")
    kpi(c4, "EXACT DUPLICATES",f"{dup_count:,}",        f"{dup_count/total*100:.1f}% copy-paste",   "#f59e0b")

    st.markdown("<br>", unsafe_allow_html=True)

    col_l, col_r = st.columns(2)
    with col_l:
        st.markdown("#### Fake vs Genuine Distribution")
        status_counts = df["fake_status"].value_counts()
        st.bar_chart(status_counts, color=["#7c3aed"])

    with col_r:
        st.markdown("#### Avg Fraud Score by Star Rating")
        if "Rate" in df.columns and "fraud_score" in df.columns:
            rating_fraud = (
                df[df["Rate"].between(1, 5)]
                .groupby("Rate")["fraud_score"]
                .mean()
                .round(3)
            )
            st.bar_chart(rating_fraud, color=["#ef4444"])

    col_l2, col_r2 = st.columns(2)
    with col_l2:
        st.markdown("#### Sentiment Label Breakdown")
        if "sentiment_label" in df.columns:
            sent_counts = df["sentiment_label"].value_counts()
            st.bar_chart(sent_counts, color=["#06b6d4"])

    with col_r2:
        st.markdown("#### Fraud Score Distribution (0–7)")
        if "fraud_score" in df.columns:
            score_counts = df["fraud_score"].sort_index().value_counts().sort_index()
            st.bar_chart(score_counts, color=["#f59e0b"])

    st.markdown("#### Average Sentiment Score by Review Status")
    if "sentiment_score" in df.columns:
        sent_by_status = df.groupby("fake_status")["sentiment_score"].mean().round(4)
        st.dataframe(
            sent_by_status.reset_index().rename(
                columns={"fake_status": "Review Status", "sentiment_score": "Avg Sentiment Score"}
            ),
            width="content",
            hide_index=True,
        )

with tab2:
    st.markdown(
        "### 🔍 Real-time Fraud Analyzer\n"
        "Enter any review summary and star rating to instantly see its fraud signals."
    )

    col_in1, col_in2 = st.columns([3, 1])
    with col_in1:
        summary_input = st.text_area(
            "📝 Review Summary",
            placeholder="Paste or type a review summary here…",
            height=140,
        )
    with col_in2:
        rating_input = st.slider("⭐ Star Rating", 1, 5, 5)
        analyse_btn  = st.button("🔍 Analyse", type="primary", width="content")

    if analyse_btn:
        if not summary_input.strip():
            st.warning("Please enter a summary first.")
        else:
            cleaned = clean_text(summary_input)
            flags = {
                "Length Flag"         : length_flag(cleaned),
                "Repetition Flag"     : repetition_flag(cleaned),
                "Generic Phrase Flag" : generic_flag(cleaned),
                "Meaningless Flag"    : meaningless_flag(cleaned),
                "Rating Mismatch Flag": rating_mismatch_flag(cleaned, rating_input),
                "Caps Ratio Flag"     : caps_ratio_flag(cleaned),
            }
            score = sum(flags.values())
            prob  = fake_probability(score, max_score=7)

            st.divider()
            res_col, flag_col = st.columns([1, 2])

            with res_col:
                st.markdown("#### Verdict")
                if score == 0:
                    st.success("✅ **Genuine**")
                elif score == 1:
                    st.warning("⚠️ **Suspicious**")
                else:
                    st.error("🚫 **Likely Fake**")

                st.markdown(f"**Fraud Score:** `{score} / 7`")
                st.progress(prob)
                st.caption(f"Fake probability: **{prob*100:.1f}%**")

                sent = round(sentiment_score(cleaned), 3)
                sent_lbl = sentiment_label(sent)
                st.metric("Sentiment Polarity", sent, delta=sent_lbl)

            with flag_col:
                st.markdown("#### Flag Breakdown")
                for flag_name, val in flags.items():
                    icon = "🔴" if val else "🟢"
                    status = "**TRIGGERED**" if val else "Clear"
                    st.markdown(f"{icon} `{flag_name}` — {status}")
                st.caption("🔴 = flag fired  |  🟢 = passed")

with tab3:
    st.markdown("### 🏭 Product-Level Fake Review Intelligence")
    st.markdown("Aggregated fake review percentage per product — useful for spotting coordinated fraud campaigns.")

    if not has_data:
        st.warning("⚠️ Run `python main.py` first to generate the dataset.")
    else:
        profile = build_reviewer_profile(df)
        top_n = st.slider("Show top N products by fake review %", 5, 50, 15)
        top   = profile.head(top_n)

        st.markdown(f"#### Top {top_n} Products by Fake Review %")
        st.dataframe(
            top[["Product_name", "total_reviews", "fake_review_count",
                 "fake_review_pct", "avg_fraud_score", "avg_sentiment", "avg_rating"]]
            .rename(columns={
                "Product_name":      "Product",
                "total_reviews":      "Total Reviews",
                "fake_review_count":  "Fake Reviews",
                "fake_review_pct":    "Fake %",
                "avg_fraud_score":    "Avg Fraud Score",
                "avg_sentiment":      "Avg Sentiment",
                "avg_rating":         "Avg Rating",
            })
            .reset_index(drop=True),
            width="content",
            hide_index=True,
        )

        chart_data = top.set_index("Product_name")["fake_review_pct"]
        st.bar_chart(chart_data, color=["#ef4444"])

        st.markdown("#### 🔎 Search a Specific Product")
        search = st.text_input("Type part of a product name:")
        if search:
            results = profile[profile["Product_name"].str.contains(search, case=False, na=False)]
            if results.empty:
                st.info("No matching products found.")
            else:
                st.dataframe(results.reset_index(drop=True), width="content", hide_index=True)

with tab4:
    st.markdown("### 🗄️ Interactive Dataset Explorer")
    st.markdown("Filter and investigate flagged reviews directly.")

    if not has_data:
        st.warning("⚠️ Run `python main.py` first to generate the dataset.")
    else:
        f1, f2, f3 = st.columns(3)
        with f1:
            status_opts = ["All"] + sorted(df["fake_status"].dropna().unique().tolist())
            sel_status  = st.selectbox("📌 Review Status", status_opts)
        with f2:
            sel_rating = st.selectbox("⭐ Star Rating", ["All", 1, 2, 3, 4, 5])
        with f3:
            if "exact_duplicate" in df.columns:
                sel_dup = st.selectbox("📋 Duplicates", ["All", "Only Duplicates", "No Duplicates"])
            else:
                sel_dup = "All"

        fdf = df.copy()
        if sel_status != "All":
            fdf = fdf[fdf["fake_status"] == sel_status]
        if sel_rating != "All":
            fdf = fdf[fdf["Rate"] == int(sel_rating)]
        if sel_dup == "Only Duplicates":
            fdf = fdf[fdf["exact_duplicate"] == True]
        elif sel_dup == "No Duplicates":
            fdf = fdf[fdf["exact_duplicate"] == False]

        st.markdown(f"**{len(fdf):,} reviews** match your filters:")
        show_cols = [c for c in [
            "fake_status", "fraud_score", "Rate", "Summary",
            "sentiment_score", "sentiment_label", "exact_duplicate", "Product_name"
        ] if c in fdf.columns]

        st.dataframe(fdf[show_cols].reset_index(drop=True),
                     width="content", height=500, hide_index=True)

        csv = fdf[show_cols].to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Download Filtered Data as CSV",
            data=csv,
            file_name="filtered_reviews.csv",
            mime="text/csv",
        )
