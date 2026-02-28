import pickle

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split


DATA_PATH = "datasets/study_data.csv"
MODEL_PATH = "model.pkl"
VECTORIZER_PATH = "vectorizer.pkl"
LABEL_MAP_PATH = "label_map.pkl"


def load_dataset(path: str) -> pd.DataFrame:
    df = pd.read_csv(path).dropna(subset=["subject", "topic", "text", "difficulty"]).copy()
    df["subject"] = df["subject"].astype(str).str.strip().str.lower()
    df["topic"] = df["topic"].astype(str).str.strip().str.lower()
    df["text"] = df["text"].astype(str).str.strip().str.lower()
    df["difficulty"] = df["difficulty"].astype(str).str.strip().str.lower()
    return df


def main() -> None:
    df = load_dataset(DATA_PATH)

    label_to_id = {"basic": 0, "advanced": 1}
    id_to_label = {v: k for k, v in label_to_id.items()}
    df = df[df["difficulty"].isin(label_to_id.keys())].copy()
    df["difficulty_label"] = df["difficulty"].map(label_to_id)

    # Include subject + topic context to reduce noisy semantic overlap.
    df["features"] = df["subject"] + " " + df["topic"] + " " + df["text"]

    X_train, X_test, y_train, y_test = train_test_split(
        df["features"],
        df["difficulty_label"],
        test_size=0.2,
        random_state=42,
        stratify=df["difficulty_label"],
    )

    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        min_df=1,
        max_df=0.95,
        sublinear_tf=True,
    )
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    model = LogisticRegression(
        max_iter=4000,
        C=2.0,
        class_weight="balanced",
        solver="lbfgs",
        random_state=42,
    )
    model.fit(X_train_vec, y_train)

    y_pred = model.predict(X_test_vec)
    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {acc:.4f}")
    print(classification_report(y_test, y_pred, target_names=["basic", "advanced"]))

    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)
    with open(VECTORIZER_PATH, "wb") as f:
        pickle.dump(vectorizer, f)
    with open(LABEL_MAP_PATH, "wb") as f:
        pickle.dump(id_to_label, f)

    print("Saved model, vectorizer, and label map.")


if __name__ == "__main__":
    main()
