🛒 Flipkart Fake Review Detection

A rule-based system to detect potentially fake product reviews using text preprocessing and heuristic pattern analysis.

The project analyzes Flipkart reviews and classifies them as Likely Fake or Genuine based on suspicious text behavior.

📂 Dataset

This project uses the Flipkart Reviews dataset from Kaggle:
🔗 https://www.kaggle.com/datasets/niraliivaghani/flipkart-dataset
⚠ The dataset is not included in this repository because it increases the size (~110MB).

How to Use the Dataset
Download the dataset from the Kaggle link above.
Extract the file flipkart_reviews.csv.
Place it inside the Data/ folder of this project.
Your structure should look like:

Data/
   flipkart_reviews.csv

How the System Works

The pipeline follows these steps:
Cleans review text (removes emojis, special characters, encoding issues)
Combines Review and Summary
Applies rule-based flags
Assigns final fake status

Fake Detection Logic

A review is marked as Likely Fake if at least two suspicious patterns are detected.
Length Rule
Less than 4 words
More than 150 words
Repetition Rule

“very” used more than 3 times

Contains excessive punctuation like !!!
Generic Phrase Rule

Detects common marketing phrases such as:
nice product
worth it
good product
highly recommend
best in class
value for money

Final decision:
Score = length_flag + repetition_flag + generic_flag
If score ≥ 2 → Likely Fake  
Else → Genuine

Example

Likely Fake
Nice product very very very very good!!!

Triggers:
Generic phrase

Repetition
→ Classified as Likely Fake

Genuine
The battery backup is good and delivery was fast.

No suspicious pattern detected
→ Classified as Genuine

How to Run

Install dependency:

pip install pandas
Run the project:
python main.py

Output

After running main.py, the output file will be generated inside:
Data/fake_review_output.csv

This file contains:
Cleaned review text
Flag columns
Final Fake_Status column

🛠 Tech Stack

Python
Pandas
Regular Expressions

This project demonstrates modular code structure, practical text preprocessing, and explainable rule-based fake review detection for real-world e-commerce data.