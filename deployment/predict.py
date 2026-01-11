import pickle
import pandas as pd

MODEL_PATH = "models/final_model.pkl"

def load_model():
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)

def predict(input_df: pd.DataFrame): 
    bundle = load_model()
    model = bundle["model"]
    scaler = bundle["scaler"]
    threshold = bundle["threshold"]
    features = bundle["features"]

    X = input_df[features]
    X_scaled = scaler.transform(X)

    prob = model.predict_proba(X_scaled)[:, 1]
    preds = (prob >= threshold).astype(int)

    return preds, prob
