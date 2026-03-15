def detect_fake(row) -> str:
    score = row["fraud_score"]

    if score >= 2:
        return "Likely Fake"
    else:
        return "Genuine"

def fake_probability(fraud_score: int, max_score: int = 7) -> float:
    return min(round(fraud_score / max_score, 4), 1.0)
