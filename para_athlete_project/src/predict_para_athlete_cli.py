import joblib
import pandas as pd

# ============================================================
# Load trained models
# ============================================================
STAMINA_MODEL_PATH = "../models/stamina_model.pkl"
FATIGUE_MODEL_PATH = "../models/fatigue_model.pkl"
INJURY_RISK_MODEL_PATH = "../models/injury_risk_model.pkl"

print("🔁 Loading models...")
stamina_model = joblib.load(STAMINA_MODEL_PATH)
fatigue_model = joblib.load(FATIGUE_MODEL_PATH)
injury_risk_model = joblib.load(INJURY_RISK_MODEL_PATH)
print("✅ Models loaded.\n")


# ============================================================
# Helper functions to get typed input safely
# ============================================================

def ask_int(prompt):
    while True:
        try:
            return int(input(prompt))
        except ValueError:
            print("❌ Please enter a valid integer.")


def ask_float(prompt):
    while True:
        try:
            return float(input(prompt))
        except ValueError:
            print("❌ Please enter a valid number.")


def ask_choice(prompt, choices):
    choices_str = "/".join(choices)
    while True:
        val = input(f"{prompt} ({choices_str}): ").strip()
        if val in choices:
            return val
        print(f"❌ Please choose one of: {choices_str}")


# ============================================================
# Main CLI for single athlete prediction
# ============================================================

def main():
    print("=== Para-Athlete Health Prediction CLI ===\n")
    print("Enter the athlete details below:\n")

    # Categorical options (same as in your generator)
    gender_choices = ["Male", "Female", "Other"]
    disability_choices = [
        "Amputation",
        "Visual Impairment",
        "Cerebral Palsy",
        "Spinal Cord Injury",
        "Intellectual Impairment",
    ]
    sport_choices = [
        "Wheelchair Racing",
        "Para Swimming",
        "Para Powerlifting",
        "Para Athletics (Track)",
        "Para Archery",
    ]

    # 👉 These must match your feature columns:
    # ['age', 'gender', 'disability_type', 'sport_type',
    #  'training_days_per_week', 'sleep_hours', 'heart_rate_rest',
    #  'daily_calorie_intake', 'protein_intake_g',
    #  'water_intake_liters', 'hydration_level']

    age = ask_int("Age (years): ")
    gender = ask_choice("Gender", gender_choices)
    disability_type = ask_choice("Disability type", disability_choices)
    sport_type = ask_choice("Sport type", sport_choices)

    training_days_per_week = ask_int("Training days per week (e.g., 3–7): ")
    sleep_hours = ask_float("Average sleep hours per night (e.g., 6.5): ")

    heart_rate_rest = ask_int("Resting heart rate (bpm): ")
    daily_calorie_intake = ask_int("Daily calorie intake (kcal): ")
    protein_intake_g = ask_float("Daily protein intake (grams): ")
    water_intake_liters = ask_float("Daily water intake (liters): ")
    hydration_level = ask_int("Hydration level (0–100): ")

    # Build a single-row DataFrame
    data = {
        "age": age,
        "gender": gender,
        "disability_type": disability_type,
        "sport_type": sport_type,
        "training_days_per_week": training_days_per_week,
        "sleep_hours": sleep_hours,
        "heart_rate_rest": heart_rate_rest,
        "daily_calorie_intake": daily_calorie_intake,
        "protein_intake_g": protein_intake_g,
        "water_intake_liters": water_intake_liters,
        "hydration_level": hydration_level,
    }

    df_input = pd.DataFrame([data])

    # Predict with each model
    stamina_pred = stamina_model.predict(df_input)[0]
    fatigue_pred = fatigue_model.predict(df_input)[0]
    injury_risk_pred = injury_risk_model.predict(df_input)[0]

    # Interpret injury risk
    if injury_risk_pred < 0.33:
        risk_label = "Low"
    elif injury_risk_pred < 0.66:
        risk_label = "Moderate"
    else:
        risk_label = "High"

    print("\n=== Prediction Results ===")
    print(f"🟢 Predicted stamina level:      {stamina_pred:.2f} (0–100)")
    print(f"🟠 Predicted fatigue level:      {fatigue_pred:.2f} (0–10)")
    print(f"🔴 Predicted injury risk score:  {injury_risk_pred:.2f} (0–1) → {risk_label} risk")
    print("\nDone.\n")


if __name__ == "__main__":
    main()

