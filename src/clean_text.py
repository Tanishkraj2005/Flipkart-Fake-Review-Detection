import re
import unicodedata

EMOJI_PATTERN = re.compile(
    "["
    u"\U0001F600-\U0001F64F"
    u"\U0001F300-\U0001F5FF"
    u"\U0001F680-\U0001F6FF"
    u"\U0001F1E0-\U0001F1FF"
    "]+",
    flags=re.UNICODE,
)

def clean_text(text: str) -> str:
    if text is None:
        return ""

    text = str(text)
    text = unicodedata.normalize("NFKD", text)
    text = text.encode("ascii", "ignore").decode("ascii")
    text = EMOJI_PATTERN.sub(" ", text)
    text = text.replace("?", " ")
    text = re.sub(r"[^a-zA-Z0-9.,!\'\"()\- ]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def clean_product_name(name: str) -> str:
    if name is None:
        return ""

    name = str(name)
    name = unicodedata.normalize("NFKD", name)
    name = name.encode("ascii", "ignore").decode("ascii")
    name = name.replace("?", " ")
    name = re.sub(r"[^a-zA-Z0-9()\-\s]", " ", name)
    name = re.sub(r"\s+", " ", name)
    return name.strip()


def caps_ratio(text: str) -> float:
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return 0.0
    return sum(1 for c in letters if c.isupper()) / len(letters)