def evaluate_repe_status(row):
    if row["Ground_Truth"] == "Malicious" and row["Prediction"] == "Malicious":
        return "True Anomaly (Detected Attack)"
    elif row["Ground_Truth"] == "Malicious" and row["Prediction"] == "Benign":
        return "Missed Attack (False Negative)"
    elif row["Ground_Truth"] == "Benign" and row["Prediction"] == "Malicious":
        return "False Alarm (False Positive)"
    return "Benign (Passed)"