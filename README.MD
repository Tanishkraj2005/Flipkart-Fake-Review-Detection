🛒 Flipkart Fake Review Detection

Detecting suspicious product reviews using rule-based NLP techniques.

This project analyzes Flipkart reviews and flags potentially fake ones based on text patterns such as unnatural length, repetitive hype words, and generic marketing phrases.

Why This Project?

Fake reviews reduce customer trust in e-commerce platforms.
This system applies heuristic-based logic to identify reviews that look suspicious — without using heavy machine learning models.

How It Works
Cleans review text (removes emojis, special characters, encoding issues)
Merges review + summary

Applies rule-based flags:

Very short or very long reviews
Excessive repetition (e.g., “very very very”)
Generic marketing phrases
Assigns a final status:

Likely Fake
Genuine

📂 Dataset

Dataset used (not included due to size ~110MB):

🔗 https://www.kaggle.com/datasets/niraliivaghani/flipkart-dataset

Download it manually and place inside the Data/ folder.

🛠 Tech Stack

Python
Pandas
Regular Expressions

▶️ Run Locally

pip install pandas
python main.py


Output will be generated inside:
Data/fake_review_output.csv



⭐ If you found this interesting, feel free to explore or extend it.