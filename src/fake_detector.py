def detect_fake(row):
    score = (
        row['length_flag'] +
        row['repetition_flag'] +
        row['generic_flag']
    )

    return "Likely Fake" if score >= 2 else "Genuine"
