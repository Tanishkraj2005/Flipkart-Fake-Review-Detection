import pandas as pd
from src.clean_text import clean_text, clean_product_name
from src.flags import length_flag, repetition_flag, generic_flag
from src.fake_detector import detect_fake

# Load dataset
df = pd.read_csv("data/flipkart_reviews.csv", encoding="latin1", on_bad_lines='skip')

# Convert Rate to numeric
df['Rate'] = pd.to_numeric(df['Rate'], errors='coerce')
df['Rate'] = df['Rate'].fillna(0)

# Clean product name
df['Product_name'] = df['Product_name'].apply(clean_product_name)

# Combine Review + Summary
df['combined_review'] = df['Review'].astype(str) + " " + df['Summary'].astype(str)

# Clean combined review
df['combined_review'] = df['combined_review'].apply(clean_text)

# Remove original columns
df = df.drop(columns=['Review', 'Summary'])

# Rename combined_review → review
df.rename(columns={'combined_review': 'review'}, inplace=True)

# Apply rules for fake review detection using 'review'
df['length_flag'] = df['review'].apply(length_flag)
df['repetition_flag'] = df['review'].apply(repetition_flag)
df['generic_flag'] = df['review'].apply(generic_flag)

# Final fake review detection
df['Fake_Status'] = df.apply(detect_fake, axis=1)

# Save output
df.to_csv("data/fake_review_output.csv", index=False)

print("✅ Fake Review Detection Completed Successfully!")
print("📁 Output saved to: data/fake_review_output.csv")
