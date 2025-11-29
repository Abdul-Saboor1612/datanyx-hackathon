import os
import joblib
import pandas as pd

# Paths to models (relative to this file: src/)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "../models")

STAMINA_MODEL_PATH = os.path.join(MODELS_DIR, "stamina_model.pkl")
FATIGUE_MODEL_PATH = os.path.join(MODELS_DIR, "fatigue_model.pkl")
INJURY_RISK_MODEL_PATH = os.path.join(MODELS_DIR, "injury_risk_model.pkl")

# Load models once at import
print("🔁 Loading para-athlete models (para_predictor.py)...")
stamina_model = joblib.load(STAMINA_MODEL_PATH)
fatigue_model = joblib.load(FATIGUE_MODEL_PATH)
injury_risk_model = joblib.load(INJURY_RISK_MODEL_PATH)
print("✅ Models loaded.\n")

# The feature columns we expect (MUST match training)
FEATURE_COLUMNS = [
    "age",
    "gender",
    "disability_type",
    "sport_type",
    "training_days_per_week",
    "sleep_hours",
    "heart_rate_rest",
    "daily_calorie_intake",
    "protein_intake_g",
    "water_intake_liters",
    "hydration_level",
]


def predict_para_athlete(features: dict) -> dict:
    """
    features: dict with keys in FEATURE_COLUMNS.
    Returns: dict with stamina, fatigue, injury_risk, risk_label.
    """

    # Ensure all needed keys exist
    missing = [col for col in FEATURE_COLUMNS if col not in features]
    if missing:
        raise ValueError(f"Missing feature keys: {missing}")

    # Build a single-row DataFrame
    df_input = pd.DataFrame([{col: features[col] for col in FEATURE_COLUMNS}])

    # Predict
    stamina_pred = stamina_model.predict(df_input)[0]
    fatigue_pred = fatigue_model.predict(df_input)[0]
    injury_risk_pred = injury_risk_model.predict(df_input)[0]

    # Label risk
    if injury_risk_pred < 0.33:
        risk_label = "Low"
    elif injury_risk_pred < 0.66:
        risk_label = "Moderate"
    else:
        risk_label = "High"

    return {
        "stamina_level": float(stamina_pred),
        "fatigue_level": float(fatigue_pred),
        "injury_risk_score": float(injury_risk_pred),
        "injury_risk_label": risk_label,
    }


if __name__ == "__main__":
    # quick manual test
    sample_input = {
        "age": 24,
        "gender": "Male",
        "disability_type": "Spinal Cord Injury",
        "sport_type": "Wheelchair Racing",
        "training_days_per_week": 6,
        "sleep_hours": 7.0,
        "heart_rate_rest": 60,
        "daily_calorie_intake": 2600,
        "protein_intake_g": 130.0,
        "water_intake_liters": 3.0,
        "hydration_level": 75,
    }

    preds = predict_para_athlete(sample_input)
    print("Test prediction:", preds)
