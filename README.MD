# 🛒 Flipkart Fake Review Detection

![Python](https://img.shields.io/badge/Python-3.x-blue.svg)
![Pandas](https://img.shields.io/badge/Pandas-Data_Analysis-green.svg)
![Machine Learning](https://img.shields.io/badge/Method-Rule__Based-orange.svg)

A robust, rule-based system designed to detect potentially fake product reviews on e-commerce platforms like Flipkart. By leveraging text preprocessing and heuristic pattern analysis, this project classifies reviews as **Likely Fake** or **Genuine** based on suspicious user behavior and language patterns.

---

## 🚀 Key Features

- **Text Preprocessing Pipeline:** Cleanses text by removing emojis, special characters, and decoding issues to ensure accurate analysis.
- **Rule-Based Detection:** Applies multiple flags based on review length, repetition, and generic marketing language.
- **Explainable Results:** Generates human-readable reasons for why a specific review was flagged.
- **Scalable Architecture:** Modular code structure (`src/` directory) for easy future ML integrations.

---

## 📂 Project Structure

```text
Flipkart-Review/
│
├── Data/
│   ├── flipkart_reviews.csv       <-- Place downloaded dataset here
│   └── fake_review_output.csv     <-- Generated output
│
├── src/
│   ├── clean_text.py              <-- Text & product name normalization
│   ├── fake_detector.py           <-- Decision logic (Likely Fake vs Genuine)
│   ├── fake_reason.py             <-- Extractor for detection explanations
│   └── flags.py                   <-- Core rule functions (length, repetition, etc.)
│
├── main.py                        <-- Entry point of the script
└── README.md                      <-- Project documentation
```

---

## 📊 Dataset

This project uses the **Flipkart Reviews dataset** from Kaggle:
🔗 [Dataset Link](https://www.kaggle.com/datasets/niraliivaghani/flipkart-dataset)

⚠️ *Note: The dataset is not included in this repository to keep the repo lightweight.*

### How to Use the Dataset
1. Download the dataset from the Kaggle link above.
2. Extract the file `flipkart_reviews.csv`.
3. Place it inside the `Data/` folder of this project.

---

## 🧠 How the System Works

The detection pipeline follows these crucial steps:

1. **Text Normalization:** Cleans the review text (strips unicode artifacts, emojis, unneeded punctuation).
2. **Feature Extraction (Flags):**
   - **Length Rule:** Flags reviews with fewer than 4 words or more than 150 words.
   - **Repetition Rule:** Identifies abnormal patterns, such as the word "very" used >3 times, or excessive punctuation (`!!!`).
   - **Generic Phrase Rule:** Detects overly common marketing phrases (e.g., "nice product", "worth it", "best in class") in short reviews.
3. **Scoring & Classification:**
   - **Score** = `length_flag` + `repetition_flag` + `generic_flag`
   - If **Score ≥ 2** ➡️ `Likely Fake`
   - Else ➡️ `Genuine`

### Examples
- **Likely Fake:** *"Nice product very very very very good!!!"*
  - **Triggers:** Generic phrase, Repetition. (Score: 2) -> `Likely Fake`
- **Genuine:** *"The battery backup is good and delivery was fast."*
  - **Triggers:** None. (Score: 0) -> `Genuine`

---

## ⚙️ Setup and Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Tanishkraj2005/Flipkart-Fake-Review-Detection.git
   cd Flipkart-Fake-Review-Detection
   ```

2. **Install dependencies:**
   Make sure you have Pandas installed in your environment.
   ```bash
   pip install pandas
   ```

3. **Run the script:**
   ```bash
   python main.py
   ```

4. **View the Output:**
   Once finished, the script saves its results to `Data/fake_review_output.csv` with the appended `Fake_Status` column.

---

## 🛠 Tech Stack

- **Language:** Python
- **Data Manipulation:** Pandas
- **String Processing:** Regular Expressions (`re`), `unicodedata`

---

## 🤝 Contributing

Contributions are welcome! If you'd like to improve the preprocessing, add machine learning classifiers (like NLP transformers or Naive Bayes), or add more heuristic rules, feel free to submit a pull request.