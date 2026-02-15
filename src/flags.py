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
    "nice product",
    "worth it",
    "worth every penny",
    "good product",
    "awesome",
    "superb",
    "value for money",
    "highly recommend",
    "best in class"
]

def generic_flag(text):
    t = text.lower()
    for w in GENERIC_WORDS:
        if w in t:
            return 1
    return 0
