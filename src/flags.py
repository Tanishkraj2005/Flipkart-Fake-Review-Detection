# 1. Length rule
def length_flag(text):
    words = len(text.split())
    return 1 if (words < 4 or words > 150) else 0




# 3. Repetition rule
def repetition_flag(text):
    text_lower = text.lower()

    # Count repetitive hype words
    if text_lower.count("very") > 3:
        return 1

    # Excessive punctuation
    if "!!!" in text:
        return 1

    return 0


# 4. Generic review rule
GENERIC_WORDS = [
    "product was good"
    "great product",
    "nice product",
    "good product",
    "very good",
    "awesome",
    "superb",
    "excellent",
    "best product",
    "best",
    "best price",
    "worth it",
    "value for money",
    "highly recommend",
    "amazing",
    "fantastic",
    "must buy",
    "good",
    "super deal",
    "ok",
    "nice",
]


def generic_flag(text):
    t = text.lower()
    word_count = len(t.split())

    for w in GENERIC_WORDS:
        if w in t and word_count <= 6:
            return 1

    return 0

