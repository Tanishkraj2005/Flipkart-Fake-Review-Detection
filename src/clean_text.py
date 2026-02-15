import re
import unicodedata

def clean_text(text):
    if text is None:
        return ""

    text = str(text)

    text = unicodedata.normalize("NFKD", text)
    text = text.encode("ascii", "ignore").decode("ascii")

    # Remove emojis
    emoji_pattern = re.compile(
        "[" 
        u"\U0001F600-\U0001F64F"
        u"\U0001F300-\U0001F5FF"
        u"\U0001F680-\U0001F6FF"
        u"\U0001F1E0-\U0001F1FF"
        "]+", flags=re.UNICODE)
    text = emoji_pattern.sub(" ", text)

    # ❌ REMOVE ALL QUESTION MARKS
    text = text.replace("?", " ")

    # Keep only allowed characters
    text = re.sub(r"[^a-zA-Z0-9.,!\'\"()\- ]", " ", text)

    text = re.sub(r"\s+", " ", text)
    return text.strip()


# Clean product name
def clean_product_name(name):
    if name is None:
        return ""

    name = str(name)

    # Fix encoding
    name = unicodedata.normalize("NFKD", name)
    name = name.encode("ascii", "ignore").decode("ascii")

    name = name.replace("?", " ")

    # Keep only letters, numbers, (), -, spaces
    name = re.sub(r"[^a-zA-Z0-9()\-\s]", " ", name)

    name = re.sub(r"\s+", " ", name)
    return name.strip()
