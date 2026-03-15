# 🛡️ Flipkart Fake Review Analytics System

<div align="center">

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io)
[![Power BI](https://img.shields.io/badge/Power%20BI-Dashboard-F2C811?style=for-the-badge&logo=powerbi&logoColor=black)](https://powerbi.microsoft.com)
[![pandas](https://img.shields.io/badge/pandas-2.0%2B-150458?style=for-the-badge&logo=pandas&logoColor=white)](https://pandas.pydata.org)
[![TextBlob](https://img.shields.io/badge/NLP-TextBlob-8A2BE2?style=for-the-badge)](https://textblob.readthedocs.io)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)

**An end-to-end data analytics pipeline that detects fake product reviews on Flipkart using rule-based fraud signals, NLP sentiment analysis, duplicate detection, and a Logistic Regression model — across 363,000+ real reviews.**

[🚀 Quick Start](#-quick-start) · [🧠 How It Works](#-how-it-works) · [📊 Live Dashboard](#-power-bi-dashboard) · [🌐 Streamlit App](#-streamlit-app) · [📁 Project Structure](#-project-structure)

</div>

---

## 📌 Overview

E-commerce platforms like Flipkart suffer from **fake, bot-generated, and incentivised reviews** that mislead buyers and distort product rankings. This project solves that with a fully automated, **transparent and explainable** analytics pipeline that:

- Processes **363,000+ raw Flipkart reviews** end-to-end
- Applies **7 independent rule-based fraud signals** to every review
- Computes a **composite fraud score** and classifies each review as `Genuine` or `Likely Fake`
- Trains a **Logistic Regression ML model** to cross-validate rule-based findings
- Detects **exact duplicate reviews** (bot copy-paste campaigns)
- Profiles every **product** by its fake review concentration
- Exports analysis-ready data for **Power BI** dashboards
- Serves a live **Streamlit app** for real-time review analysis

> **Why explainable?** Every fake label traces directly back to which specific signals triggered it — not a black-box model.

---

## 📁 Project Structure

```
flipkart_fake_review/
│
├── main.py                        # 🔧 Master pipeline — run this first
│
├── src/                           # Core analytics modules
│   ├── __init__.py
│   ├── clean_text.py              # Text cleaning: unicode, emoji, encoding artifacts
│   ├── feature_engineering.py    # Feature pipeline orchestrator
│   ├── flags.py                   # 7 rule-based fraud detection signals
│   ├── fake_detector.py           # Fraud score → Genuine / Likely Fake label
│   ├── sentiment.py               # TextBlob polarity & sentiment label
│   ├── duplicate_detection.py     # Exact duplicate detection via value_counts
│   ├── reviewer_detection.py      # Product-level fake review profiling
│   └── ml_model.py                # Logistic Regression model training & evaluation
│
├── app/
│   └── app.py                     # 🌐 Streamlit app (4 tabs, full analytics UI)
│
├── analysis/
│   └── eda_analysis.ipynb         # 📓 Jupyter EDA: 10+ charts & statistical insights
│
├── Data/
│   ├── flipkart_reviews.csv       # Raw dataset (download from Kaggle — see below)
│   └── processed_reviews.csv      # ✅ Generated output (after running main.py)
│
├── dashboard_data/
│   ├── powerbi_dataset.csv        # Power BI export (after running main.py)
│   └── Fraud Detection.pbix       # Power BI report file
│
├── requirements.txt               # Python dependencies
├── pyrightconfig.json             # Type-checking config
└── .gitignore
```

---

## 🚀 Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/flipkart-fake-review-detector.git
cd flipkart-fake-review-detector
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
python -m textblob.download_corpora
```

### 3. Add the dataset

Download `flipkart_reviews.csv` from the [Kaggle Dataset](https://www.kaggle.com/datasets/niraliivaghani/flipkart-dataset) and place it inside the `Data/` folder.

```
Data/
└── flipkart_reviews.csv   ← place here
```

### 4. Run the full pipeline

```bash
python main.py
```

This runs all 9 pipeline steps (~3–5 minutes) and generates:
- `Data/processed_reviews.csv` — full output with all features, flags, scores, and labels
- `dashboard_data/powerbi_dataset.csv` — trimmed Power BI–ready export

### 5. Launch the Streamlit app

```bash
streamlit run app/app.py
```

### 6. Explore the EDA notebook

```bash
jupyter notebook analysis/eda_analysis.ipynb
```

---

## 🧠 How It Works

The pipeline runs in **9 sequential steps**, each clearly logged to the console:

```
[1/9] Load raw CSV
[2/9] Clean & normalise text
[3/9] Engineer text features
[4/9] Apply rule-based fraud detection
[5/9] Train Logistic Regression ML model
[6/9] Detect exact duplicates
[7/9] Compute sentiment labels
[8/9] Save processed_reviews.csv
[9/9] Export powerbi_dataset.csv
```

### Step 1 — Data Loading (`main.py`)

Reads the raw CSV with `latin1` encoding (handles Flipkart's special characters) and skips malformed rows gracefully.

```python
df = pd.read_csv(path, encoding="latin1", on_bad_lines="skip", low_memory=False)
```

### Step 2 — Text Cleaning (`src/clean_text.py`)

Every review and product name goes through a multi-stage cleaner:

| Step | What it does |
|------|-------------|
| Unicode normalisation | Converts fancy Unicode to ASCII equivalents |
| Emoji removal | Strips emoji characters using a Unicode range regex |
| Encoding artifacts | Removes `?` and non-printable characters |
| Whitespace collapse | Normalises multiple spaces to single space |

```python
clean_text("Amazing product!! 🔥 Super quality??")
# output: "Amazing product!! Super quality"
```

Also handles `Rate` (converts non-numeric to median) and `Price` (strips commas and currency symbols).

### Step 3 — Feature Engineering (`src/feature_engineering.py`)

Builds structured numeric features from raw text for both rule-based and ML analysis:

| Feature | Description |
|---------|-------------|
| `review_length` | Character count of the cleaned Summary |
| `word_count` | Word count of the Summary |
| `caps_ratio` | Fraction of alphabetic characters that are UPPERCASE |
| `avg_word_length` | Average characters per word |
| `sentence_count` | Number of sentences (split on `.`, `!`, `?`) |
| `exclamation_count` | Number of `!` in the summary |
| `sentiment_score` | TextBlob polarity score from −1.0 to +1.0 |

### Step 4 — Rule-Based Fraud Detection (`src/flags.py`)

The heart of the system. Seven independent binary signals are applied to every review:

| # | Signal | Trigger Condition | Column |
|---|--------|-------------------|--------|
| 1 | `length_flag` | Summary exceeds 150 words (keyword stuffing) | `Summary` |
| 2 | `repetition_flag` | A word repeated 10+ times, or >30% of total words; or `!!!` / `???` | `Summary` |
| 3 | `generic_flag` | Entire summary matches a known lazy phrase (e.g. `"great product"`, `"osm"`, `"value for money"`) | `Summary` |
| 4 | `rating_mismatch_flag` | Negative words + rating ≥ 4, or positive words + rating ≤ 2 | `Summary` + `Rate` |
| 5 | `meaningless_flag` | Placeholder text: `"ok"`, `"hi"`, `"na"`, `"test"`, single/double chars | `Summary` |
| 6 | `duplicate_flag` | Same summary appears 3+ times across the entire dataset | `Summary` |
| 7 | `caps_ratio_flag` | More than 40% of alphabetic characters are UPPERCASE | `Summary` |

**Why the `Summary` column?**  
The Summary is the review headline — the most concise signal of intent. It's the field bots abuse most (single-word, copy-pasted, or generic). The full Review body naturally repeats words, making rule-based signals far too noisy there.

### Step 4b — Fraud Score & Label (`src/fake_detector.py`)

```
fraud_score = sum of all 7 binary flags   →   range: 0 to 7

fraud_score == 0   →  ✅ Genuine
fraud_score == 1   →  ⚠️  Borderline (treated as Genuine)
fraud_score >= 2   →  🔴 Likely Fake
```

The `fake_probability` function normalises the score to a 0–1 float for the Streamlit probability bar.

### Step 5 — ML Model (`src/ml_model.py`)

A **Logistic Regression** model is trained on four clean numeric features:

| Feature Used | Why |
|-------------|-----|
| `review_length` | Bots tend to write very short or very long reviews |
| `word_count` | Complements length |
| `caps_ratio` | Excessive caps is a spam signal |
| `sentiment_score` | Fake reviews tend to be abnormally positive |

```
80/20 train-test split → random_state=42 for reproducibility
Outputs: accuracy, confusion matrix, classification report, feature importances
Adds `ml_prediction` column to the dataset
```

### Step 6 — Exact Duplicate Detection (`src/duplicate_detection.py`)

Uses a `value_counts` approach — if an exact summary string appears 2+ times in the dataset, it's tagged `exact_duplicate = True`. This catches coordinated bot campaigns where the same text is submitted under multiple accounts.

### Step 7 — Sentiment Labels (`src/sentiment.py`)

Converts the numeric `sentiment_score` to a human-readable label:

```
score >  0.05  →  Positive
score < -0.05  →  Negative
otherwise      →  Neutral
```

---

## 📊 Output Columns

The processed CSV and Power BI export contain the following columns:

| Column | Type | Description |
|--------|------|-------------|
| `Product_name` | `str` | Cleaned product name |
| `Price` | `float` | Product price (cleaned) |
| `Rate` | `float` | Star rating (1–5) |
| `Review` | `str` | Full cleaned review body |
| `Summary` | `str` | Cleaned review headline |
| `review_length` | `int` | Character count of Summary |
| `word_count` | `int` | Word count of Summary |
| `caps_ratio` | `float` | Fraction of uppercase letters |
| `avg_word_length` | `float` | Average chars per word |
| `sentiment_score` | `float` | TextBlob polarity [−1.0, +1.0] |
| `sentiment_label` | `str` | Positive / Neutral / Negative |
| `length_flag` | `int` | 0 or 1 |
| `repetition_flag` | `int` | 0 or 1 |
| `generic_flag` | `int` | 0 or 1 |
| `rating_mismatch_flag` | `int` | 0 or 1 |
| `duplicate_flag` | `int` | 0 or 1 |
| `meaningless_flag` | `int` | 0 or 1 |
| `caps_ratio_flag` | `int` | 0 or 1 |
| `exact_duplicate` | `bool` | True if summary seen 2+ times |
| `fraud_score` | `int` | Sum of all 7 flags (0–7) |
| `fake_status` | `str` | `Genuine` / `Likely Fake` |
| `ml_prediction` | `str` | `Genuine` / `Likely Fake` (from ML model) |

---

## 🌐 Streamlit App

The app (`app/app.py`) provides a **4-tab interactive analytics interface**:

| Tab | What You Get |
|-----|-------------|
| **📊 Dashboard** | KPI cards (total reviews, fake %, genuine %, duplicates). Charts: status distribution, fraud score by rating, sentiment breakdown |
| **🔍 Review Analyzer** | Paste any summary text → instant fraud score, flag-by-flag breakdown, TextBlob sentiment polarity, fake probability bar |
| **🏭 Product Intelligence** | Top-N products ranked by fake review %. Searchable product lookup. Bar chart of worst offenders |
| **🗄️ Dataset Explorer** | Filter by status, rating, duplicates. Download filtered CSV directly from the app |

```bash
streamlit run app/app.py
```

---

## 📊 Power BI Dashboard

Explore the interactive analytics dashboard:

🔗 **[View Live Dashboard on Power BI](https://app.powerbi.com/groups/me/reports/bd7903d9-a010-4c77-a7e9-93631bb3abdd/c6451f46491c018d3ad2?experience=power-bi)**

The dashboard (`dashboard_data/Fraud Detection.pbix`) is built on `powerbi_dataset.csv` and visualises:
- Fake vs Genuine review distribution
- Fraud score heatmap by star rating
- Top products with highest fake review concentration
- Sentiment trend across rating categories

---

## 📈 Key Findings (EDA)

From `analysis/eda_analysis.ipynb`:

- **~30–35%** of reviews are flagged as `Likely Fake`
- **~64%** of all reviews have exact duplicate summaries — clear evidence of coordinated bot activity
- **1-star and 5-star** reviews have the highest average fraud scores (common in review manipulation campaigns)
- **Fake reviews** carry significantly higher average sentiment scores — bots write unnaturally positive text
- Several products show **80–100% fake review rates** — indicating listing manipulation

---

## 🛠️ Tech Stack

| Layer | Tool | Purpose |
|-------|------|---------|
| Language | Python 3.10+ | Core pipeline |
| Data Processing | pandas 2.0+ | ETL, cleaning, feature engineering |
| NLP | TextBlob | Sentiment polarity scoring |
| Machine Learning | scikit-learn | Logistic Regression model |
| Visualisation | matplotlib, seaborn | EDA charts in Jupyter |
| Web App | Streamlit | Interactive analytics UI |
| Dashboard | Power BI | Business-level reporting |
| Notebook | Jupyter | Exploratory Data Analysis |

---

## 💼 Business Value

| Use Case | How This Project Helps |
|----------|------------------------|
| **Platform moderation** | Auto-flag low-quality reviews before they go live |
| **Competitor analysis** | Identify products with artificially inflated ratings |
| **Consumer trust** | Surface genuine reviews and suppress spam |
| **Review bombing detection** | Spot coordinated negative / positive campaigns |
| **Seller compliance** | Audit trail of flagged reviews per product |

---

## 📦 Dependencies

```txt
pandas>=2.0.0
textblob>=0.17.1
streamlit>=1.32.0
matplotlib>=3.7.0
seaborn>=0.13.0
wordcloud>=1.9.0
nltk>=3.8.0
scipy>=1.11.0
scikit-learn
```

Install all at once:

```bash
pip install -r requirements.txt
python -m textblob.download_corpora
```

---

## 👨‍💻 About This Project

This is a Data Analyst portfolio project. It covers the full workflow — raw data cleaning, rule-based logic, NLP feature engineering, ML modelling, EDA in Jupyter, a Streamlit app, and a Power BI dashboard.

---

## 📄 License

MIT License — free to use, modify, and distribute.
