import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report

def run_ml_model(df):

    print("\n[ML] Training Machine Learning Model...")

    features = [
        "review_length",
        "word_count",
        "caps_ratio",
        "sentiment_score"
    ]

    X = df[features].fillna(0)
    y = (df["fake_status"] == "Likely Fake").astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    accuracy = model.score(X_test, y_test)
    print(f"✓ ML Model Accuracy: {accuracy:.2f}")

    preds = model.predict(X_test)

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, preds))

    print("\nClassification Report:")
    print(classification_report(y_test, preds))

    importance = pd.DataFrame({
        "Feature": features,
        "Importance": model.coef_[0]
    }).sort_values(by="Importance", ascending=False)

    print("\nFeature Importance:")
    print(importance)

    df["ml_prediction"] = model.predict(X)
    df["ml_prediction"] = df["ml_prediction"].map({
        0: "Genuine",
        1: "Likely Fake"
    })
    return df