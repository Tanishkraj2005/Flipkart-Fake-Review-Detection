import re

def length_flag(text: str) -> int:
    words = len(text.split())
    if words > 150:
        return 1
    return 0

def repetition_flag(text: str) -> int:
    t = text.lower()
    if t.count("very") > 3 or t.count("really") > 3 or t.count("super") > 3:
        return 1
    if "!!!" in text or "???" in text:
        return 1
    words = t.split()
    total_words = len(words)
    for word in set(words):
        if len(word) > 2 and (words.count(word) >= 10 or (total_words > 10 and words.count(word)/total_words > 0.3)):
            return 1
    return 0

GENERIC_PHRASES = {
    "great product", "nice product", "good product",
    "awesome product", "superb product", "excellent product",
    "best product", "worst product", "amazing product",
    "fantastic product", "perfect product", "good quality",
    "nice quality", "very good", "not good",
    "product was good", "best price", "worth it",
    "value for money", "highly recommend", "must buy",
    "super deal", "five stars", "5 stars",
    "osm", "awsm", "nyc",
}

def generic_flag(text: str) -> int:
    t = text.lower().strip()
    return 1 if t in GENERIC_PHRASES else 0

def rating_mismatch_flag(summary: str, rating: float) -> int:
    t = str(summary).lower()
    negative_keywords = ["worst", "terrible", "horrible", "pathetic", "disaster"]
    if rating >= 4 and any(kw in t for kw in negative_keywords):
        return 1
    positive_keywords = ["awesome", "excellent", "superb", "best", "great", "perfect"]
    if rating <= 2 and any(kw in t for kw in positive_keywords):
        return 1
    return 0

NON_REVIEW_TOKENS = {
    "hello", "hi", "thanks", "thank you", "ok", "okay", "test",
    "na", "n/a", "nil", "none", "no review", "no comment",
    "fine", ".", "..", "...", "good", "bad",
}

def meaningless_flag(text: str) -> int:
    normalized = text.lower().strip()
    if normalized in NON_REVIEW_TOKENS:
        return 1
    if len(normalized) <= 2:
        return 1
    return 0

def duplicate_flag(text: str, review_counts: dict) -> int:
    text_lower = text.lower().strip()
    if review_counts.get(text_lower, 0) > 2:
        return 1
    return 0

def caps_ratio_flag(text: str, threshold: float = 0.4) -> int:
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return 0
    ratio = sum(1 for c in letters if c.isupper()) / len(letters)
    return 1 if ratio > threshold else 0

def short_review_flag(text: str, min_words: int = 4) -> int:
    words = [w for w in text.strip().split() if w]
    return 1 if len(words) <= min_words else 0
