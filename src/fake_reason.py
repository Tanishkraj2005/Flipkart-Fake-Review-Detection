def get_fake_reason(row):
    reasons = []

    if row['length_flag'] == 1:
        reasons.append("Unnatural length")

    if row['repetition_flag'] == 1:
        reasons.append("Repetitive hype words")

    if row['generic_flag'] == 1:
        reasons.append("Generic marketing phrases")

    if len(reasons) == 0:
        return "No suspicious pattern"

    return ", ".join(reasons)
