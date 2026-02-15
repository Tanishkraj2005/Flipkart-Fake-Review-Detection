Flipkart Fake Review Detection

This project detects potentially fake reviews from a Flipkart dataset using rule-based logic.
The system cleans review text and applies simple behavioral rules to classify reviews as Likely Fake or Genuine.

Dataset:

Dataset source (not included in this repo due to size ~110MB):

https://www.kaggle.com/datasets/niraliivaghani/flipkart-dataset

Please download the dataset manually and place flipkart_reviews.csv inside the data/ folder.

What This Project Does

Cleans review text (removes emojis, special characters, encoding issues)
Combines review and summary columns
Applies rule-based flags:
Very short or very long reviews
Excessive repetition or hype punctuation
Generic marketing phrases (e.g., "nice product", "worth it")
Classifies reviews based on a scoring system
If two or more suspicious patterns are found → Likely Fake
Otherwise → Genuine

Project Structure
data/
   flipkart_reviews.csv
   fake_review_output.csv

src/
   clean_text.py
   flags.py
   fake_detector.py
   flag_reason.py

main.py


How to Run

Install dependency:
pip install pandas


Download dataset from Kaggle and place it inside data/.

Run:
python main.py


Output will be saved as:
data/fake_review_output.csv

Tech Used
Python
Pandas
Regular Expressions