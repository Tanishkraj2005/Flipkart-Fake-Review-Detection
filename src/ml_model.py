import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix, classification_report


def run_ml_model(df):

    print("\n[ML] Training Machine Learning Model...")

    features = [
        "review_length",
        "word_count",
        "lexical_diversity",
        "caps_ratio",
        "sentiment_score"
    ]

    df["Summary"] = df["Summary"].fillna("")
    X = df[features + ["Summary"]].copy()
    for col in features:
        X[col] = pd.to_numeric(X[col], errors='coerce').fillna(0)

    y = (df["fake_status"] == "Likely Fake").astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', 'passthrough', features),
            ('text', TfidfVectorizer(max_features=500, stop_words='english'), 'Summary')
        ]
    )

    model = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1))
    ])

    model.fit(X_train, y_train)

    accuracy = model.score(X_test, y_test)
    print(f"✓ ML Model Accuracy: {accuracy:.2f}")

    preds = model.predict(X_test)

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, preds))

    print("\nClassification Report:")
    print(classification_report(y_test, preds))

    if hasattr(model.named_steps['classifier'], 'feature_importances_'):
        pass # Feature importances for pipelines with text require complex extraction, safe to skip for printing here


    df["ml_prediction"] = model.predict(X)
    df["ml_prediction"] = df["ml_prediction"].map({
        0: "Genuine",
        1: "Likely Fake"
    })
    return df