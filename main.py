import pandas as pd
from src.clean_text import clean_text, clean_product_name
from src.flags import length_flag, repetition_flag, generic_flag
from src.fake_detector import detect_fake

df = pd.read_csv("Data/flipkart_reviews.csv", encoding="latin1", on_bad_lines='skip')

df['Rate'] = pd.to_numeric(df['Rate'], errors='coerce')
df['Rate'] = df['Rate'].fillna(0)

df['Product_name'] = df['Product_name'].apply(clean_product_name)

df['Price'] = df['Price'].astype(str)
df['Price'] = df['Price'].str.replace('?', '', regex=False)
df['Price'] = df['Price'].str.strip()

df['Review'] = df['Review'].apply(clean_text)
df['Summary'] = df['Summary'].apply(clean_text)

df['length_flag'] = df['Summary'].apply(length_flag)
df['repetition_flag'] = df['Summary'].apply(repetition_flag)
df['generic_flag'] = df['Summary'].apply(generic_flag)

df['Fake_Status'] = df.apply(detect_fake, axis=1)

df.to_csv("Data/fake_review_output.csv", index=False)

print("Fake Review Detection Completed Successfully!")
print("Output saved to: Data/fake_review_output.csv")
