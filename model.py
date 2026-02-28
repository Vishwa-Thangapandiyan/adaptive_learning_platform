import joblib
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model_path_primary = os.path.join(BASE_DIR, "difficulty_model.pkl")
model_path_fallback = os.path.join(BASE_DIR, "model.pkl")

if os.path.exists(model_path_primary):
    model = joblib.load(model_path_primary)
else:
    model = joblib.load(model_path_fallback)

vectorizer = joblib.load(os.path.join(BASE_DIR, "vectorizer.pkl"))
label_map_path = os.path.join(BASE_DIR, "label_map.pkl")
label_map = joblib.load(label_map_path) if os.path.exists(label_map_path) else None

def predict_difficulty(text):
    X = vectorizer.transform([text])
    prediction = model.predict(X)
    pred = prediction[0]

    if label_map is not None:
        mapped = label_map.get(int(pred), pred) if isinstance(pred, (int, float)) else label_map.get(pred, pred)
        return str(mapped).title()

    return "Hard" if int(pred) == 1 else "Easy"
