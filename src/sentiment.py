from textblob import TextBlob
def sentiment_score(text: str) -> float:

    try:
        if not text or not text.strip():
            return 0.0
        return round(TextBlob(text).sentiment.polarity, 4)

    except Exception:
        return 0.0

def sentiment_label(score: float) -> str:
    
    if score > 0.05:
        return "Positive"
    elif score < -0.05:
        return "Negative"
    else:
        return "Neutral"


def subjectivity_score(text: str) -> float:

    try:
        if not text or not text.strip():
            return 0.0

        return round(TextBlob(text).sentiment.subjectivity, 4)
    except Exception:
        return 0.0
