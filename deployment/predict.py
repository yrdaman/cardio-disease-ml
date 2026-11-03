import pickle
import sys
import pandas as pd
import sklearn._loss.loss as sklearn_loss

MODEL_PATH = "models/final_model.pkl"

def load_model():
    sys.modules.setdefault("_loss", sklearn_loss)
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
