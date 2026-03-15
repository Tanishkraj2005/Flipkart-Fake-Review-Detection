# 🛡️ Flipkart Fake Review Analytics System

<div align="center">

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://flipkart-fake-review-detection-08.streamlit.app/)
[![Power BI](https://img.shields.io/badge/Power%20BI-Dashboard-F2C811?style=for-the-badge&logo=powerbi&logoColor=black)](https://powerbi.microsoft.com)
[![pandas](https://img.shields.io/badge/pandas-2.0%2B-150458?style=for-the-badge&logo=pandas&logoColor=white)](https://pandas.pydata.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)

**An end-to-end analytics pipeline that detects fake Flipkart product reviews using rule-based fraud signals, NLP sentiment analysis, and a Logistic Regression model — across 363,000+ real reviews.**

---

### 🔗 Quick Links
- **🌐 [Live Streamlit App](https://flipkart-fake-review-detection-08.streamlit.app/)** — Test the real-time review analyzer yourself
- **📊 [Power BI Dashboard](https://app.powerbi.com/groups/me/reports/bd7903d9-a010-4c77-a7e9-93631bb3abdd/c6451f46491c018d3ad2?experience=power-bi)** — Explore the business intelligence report
- **🗄️ [Raw Dataset (Kaggle)](https://www.kaggle.com/datasets/niraliivaghani/flipkart-dataset)** — Download the original 363K reviews

</div>

---

## 📌 Overview

Fake and bot-generated reviews mislead buyers and distort product rankings. This project builds a fully automated, **explainable** fraud detection system that processes **363,000+ raw Flipkart reviews** and classifies each one as `Genuine` or `Likely Fake` using 7 independent fraud signals, NLP sentiment scoring, and a machine learning model.

All outputs are exported to a **Streamlit app** for live analysis and a **Power BI dashboard** for business-level reporting.

---

## 📁 Project Structure

```
flipkart_fake_review/
├── main.py                     # Master pipeline — run this first
├── src/
│   ├── clean_text.py           # Unicode, emoji, encoding cleanup
│   ├── feature_engineering.py  # Text feature builder
│   ├── flags.py                # 7 fraud signal functions
│   ├── fake_detector.py        # Fraud score → label
│   ├── sentiment.py            # TextBlob polarity scoring
│   ├── duplicate_detection.py  # Exact duplicate tagging
│   └── ml_model.py             # Logistic Regression training & evaluation
├── app/
│   └── app.py                  # Streamlit app (4 tabs)
├── analysis/
│   └── eda_analysis.ipynb      # EDA notebook (10+ charts)
├── Data/
│   ├── flipkart_reviews.csv    # Raw dataset (add from Kaggle)
│   └── processed_reviews.csv   # Output of main.py
└── dashboard_data/
    ├── powerbi_dataset.csv     # Power BI export
    └── Fraud Detection.pbix    # Power BI report file
```

---

## 🔬 Module Breakdown

Every source file inside `src/` has a single, focused responsibility. Here's what each one does:

### `clean_text.py`
Handles all raw text sanitisation before any analysis runs.
- **`clean_text(text)`** — strips emojis, normalises Unicode to ASCII, removes encoding artifacts (`?`, special chars), and collapses extra whitespace. Applied to both `Review` and `Summary` columns.
- **`clean_product_name(name)`** — same cleanup but tailored for product names (preserves hyphens and brackets).
- **`caps_ratio(text)`** — calculates the fraction of alphabetic characters that are uppercase. Used as a spam signal.

### `feature_engineering.py`
Orchestrates all feature creation and calls everything in the right order.
- **`add_basic_features(df)`** — adds `review_length`, `word_count`, `caps_ratio`, `avg_word_length`, `sentence_count`, and `exclamation_count` from the `Summary` column.
- **`add_sentiment(df)`** — runs TextBlob on every summary to produce a `sentiment_score` from −1.0 to +1.0.
- **`add_duplicate_flag(df)`** — builds a frequency map of all summaries and flags any that appear 3+ times.
- **`add_rule_flags(df)`** — applies all 7 fraud signal functions from `flags.py`.
- **`add_fraud_score(df)`** — sums all 7 binary flags into a single `fraud_score` (0–7).
- **`build_features(df)`** — master function that calls all of the above in sequence.

### `flags.py`
Contains the 7 individual fraud signal functions — the core logic of the system.
- **`length_flag(text)`** — returns `1` if the summary has more than 150 words.
- **`repetition_flag(text)`** — returns `1` if a word is repeated excessively or `!!!` / `???` is present.
- **`generic_flag(text)`** — returns `1` if the entire summary matches a known lazy phrase like `"great product"` or `"value for money"`.
- **`rating_mismatch_flag(summary, rating)`** — returns `1` when the sentiment of words contradicts the star rating (e.g. negative words + 5 stars).
- **`meaningless_flag(text)`** — returns `1` for placeholder text like `"ok"`, `"hi"`, `"na"`, or single characters.
- **`duplicate_flag(text, review_counts)`** — returns `1` if the same summary appears 3+ times across the dataset.
- **`caps_ratio_flag(text)`** — returns `1` if more than 40% of letters are uppercase.

### `fake_detector.py`
Converts the numeric fraud score into a human-readable label.
- **`detect_fake(row)`** — reads `fraud_score` and returns `"Likely Fake"` if ≥ 2, else `"Genuine"`.
- **`fake_probability(fraud_score)`** — normalises the score to a 0–1 float (used by the Streamlit probability bar).

### `sentiment.py`
Wraps TextBlob for sentiment analysis.
- **`sentiment_score(text)`** — returns a polarity float from −1.0 (very negative) to +1.0 (very positive).
- **`sentiment_label(score)`** — converts the float to `"Positive"`, `"Neutral"`, or `"Negative"` using ±0.05 thresholds.
- **`subjectivity_score(text)`** — returns how opinion-based a text is (0 = factual, 1 = very subjective).

### `duplicate_detection.py`
Handles exact duplicate detection at the dataset level.
- **`flag_exact_duplicates(df)`** — uses `value_counts()` to mark any summary appearing 2+ times as `exact_duplicate = True`.
- **`duplicate_summary(df)`** — returns a summary dict with total duplicate count and percentage for reporting.

### `reviewer_detection.py`
Builds a product-level fraud profile by aggregating review data.
- **`build_reviewer_profile(df)`** — groups by `Product_name` and computes `total_reviews`, `fake_review_count`, `fake_review_pct`, `avg_fraud_score`, `avg_sentiment`, and `avg_rating` per product. Results are sorted by fake percentage — the worst offenders appear first.

### `ml_model.py`
Trains and evaluates a machine learning model to cross-validate rule-based labels.
- **`run_ml_model(df)`** — trains a **Logistic Regression** model on `review_length`, `word_count`, `caps_ratio`, and `sentiment_score` with an 80/20 train-test split. Prints accuracy, confusion matrix, classification report, and feature importance. Adds `ml_prediction` to the dataset.

---

## 🚀 Quick Start

**1. Clone & install**
```bash
git clone https://github.com/yourusername/flipkart-fake-review-detector.git
cd flipkart-fake-review-detector
pip install -r requirements.txt
python -m textblob.download_corpora
```

**2. Add the dataset**

Download `flipkart_reviews.csv` from [Kaggle](https://www.kaggle.com/datasets/niraliivaghani/flipkart-dataset) and place it in `Data/`.

**3. Run the pipeline**
```bash
python main.py
```
Processes all 363K reviews in ~3–5 minutes and exports `processed_reviews.csv` and `powerbi_dataset.csv`.

**4. Launch the app**
```bash
streamlit run app/app.py
```

**5. Explore EDA**
```bash
jupyter notebook analysis/eda_analysis.ipynb
```

---

## 🧠 Fraud Detection Logic

Every review's `Summary` field is passed through **7 binary fraud signals** (each returns `0` or `1`):

| Signal | Triggers when… |
|--------|----------------|
| `length_flag` | Summary exceeds 150 words |
| `repetition_flag` | A word appears 10+ times or makes up >30% of text |
| `generic_flag` | Summary matches a known lazy phrase (`"great product"`, `"osm"`, etc.) |
| `rating_mismatch_flag` | Negative words with high rating, or positive words with low rating |
| `meaningless_flag` | Placeholder text: `"ok"`, `"hi"`, `"na"`, single chars |
| `duplicate_flag` | Same summary appears 3+ times in the dataset |
| `caps_ratio_flag` | More than 40% of letters are UPPERCASE |

**Why `Summary` and not `Review`?** The summary is the headline — the most concise signal of intent and the field bots abuse most. The full review body naturally repeats words, making it too noisy for rule-based analysis.

### Scoring

```
fraud_score = sum of all 7 flags    (range: 0 – 7)

0     →  ✅ Genuine
1     →  Borderline (treated as Genuine)
≥ 2   →  🔴 Likely Fake
```

A **Logistic Regression** model is also trained on `review_length`, `word_count`, `caps_ratio`, and `sentiment_score` to produce an independent `ml_prediction` for cross-validation.

---

## 📊 Output Columns

| Column | Description |
|--------|-------------|
| `Product_name`, `Rate`, `Price` | cleaned product metadata |
| `Review`, `Summary` | cleaned text fields |
| `review_length`, `word_count`, `caps_ratio`, `avg_word_length` | text features |
| `sentiment_score`, `sentiment_label` | TextBlob polarity (−1.0 to +1.0) |
| `length_flag` … `caps_ratio_flag` | 7 binary fraud signals |
| `exact_duplicate` | True if summary seen 2+ times |
| `fraud_score` | Sum of all 7 flags (0–7) |
| `fake_status` | `Genuine` / `Likely Fake` |
| `ml_prediction` | Logistic Regression label |

---

## 🌐 Streamlit App

🔗 **[View Live Streamlit Web App](https://flipkart-fake-review-detection-08.streamlit.app/)**

Four-tab interactive interface:

| Tab | What You Get |
|-----|-------------|
| **📊 Dashboard** | KPI cards, status distribution, fraud score by rating, sentiment charts |
| **🔍 Review Analyzer** | Paste any summary → instant fraud score, flag breakdown, sentiment, fake probability |
| **🏭 Product Intelligence** | Products ranked by fake review %. Bar chart of worst offenders |
| **🗄️ Dataset Explorer** | Filter by status / rating / duplicates. Download filtered CSV |

---

## 📊 Power BI Dashboard

🔗 **[View Live Dashboard on Power BI](https://app.powerbi.com/groups/me/reports/bd7903d9-a010-4c77-a7e9-93631bb3abdd/c6451f46491c018d3ad2?experience=power-bi)**

The `.pbix` file is in `dashboard_data/` — open it in Power BI Desktop or view it live via the link above.

---

## 📈 Key Findings (EDA)

- **~30–35%** of reviews are flagged as `Likely Fake`
- **~64%** of all reviews have exact duplicate summaries — coordinated bot activity
- **1-star and 5-star** reviews have the highest average fraud scores
- **Fake reviews** score significantly higher on sentiment — bots write unnaturally positive text
- Several products show **80–100% fake review rates**, indicating listing manipulation

---

## 🛠️ Tech Stack

| Tool | Role |
|------|------|
| Python 3.10+, pandas | Data pipeline & processing |
| TextBlob, NLTK | NLP & sentiment scoring |
| scikit-learn | Logistic Regression model |
| matplotlib, seaborn | EDA visualisations |
| Streamlit | Interactive web app |
| Power BI | Business dashboard |
| Jupyter | Exploratory Data Analysis |

---

## 💼 Business Value

| Use Case | How This Helps |
|----------|----------------|
| **Platform moderation** | Auto-flag low-quality reviews before publication |
| **Competitor analysis** | Identify products with artificially inflated ratings |
| **Review bombing detection** | Spot coordinated negative / positive campaigns |
| **Seller compliance** | Audit trail of flagged reviews per product |

---

## 👨‍💻 About

This is a Data Analyst portfolio project covering the full workflow — data cleaning, rule-based fraud logic, NLP feature engineering, ML modelling, EDA, a Streamlit app, and a Power BI dashboard.

---

## 📄 License

MIT License — free to use, modify, and distribute.
