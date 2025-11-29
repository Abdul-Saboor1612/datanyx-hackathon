import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import joblib

# ============================================
# 1. Load dataset
# ============================================
CSV_PATH = "../data/para_athlete_synthetic_data.csv"  # change if your file name is different

df = pd.read_csv(CSV_PATH)

print("✅ Loaded dataset")
print("Shape:", df.shape)
print("\n📌 Columns:")
print(df.columns.tolist())

print("\n🔍 First 5 rows:")
print(df.head())

# ============================================
# 2. Define features & targets
# ============================================

# Targets we want to model
TARGET_COLUMNS = {
    "stamina_level": "stamina_model.pkl",
    "fatigue_level": "fatigue_model.pkl",
    "injury_risk_score": "injury_risk_model.pkl",
}

# Columns we do NOT want to use as features
EXCLUDE_AS_FEATURES = list(TARGET_COLUMNS.keys()) + [
    "performance_score",
    "overtraining_alert",  # ❗ we won't know this at prediction time
]

# Use all other columns as features
feature_cols = [col for col in df.columns if col not in EXCLUDE_AS_FEATURES]

# Categorical and numeric columns
categorical_cols = ["gender", "disability_type", "sport_type"]
numeric_cols = [col for col in feature_cols if col not in categorical_cols]

print("\n🧱 Feature columns:", feature_cols)
print("   🔤 Categorical:", categorical_cols)
print("   🔢 Numeric:", numeric_cols)

X = df[feature_cols]

# ============================================
# 3. Preprocessing & model definition
# ============================================

preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
        ("num", StandardScaler(), numeric_cols),
    ]
)

def train_and_save_model(target_name: str, model_filename: str):
    print("\n" + "=" * 60)
    print(f"🚀 Training model for target: {target_name}")
    print("=" * 60)

    y = df[target_name]

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Pipeline: preprocess → RandomForestRegressor
    model = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            (
                "regressor",
                RandomForestRegressor(
                    n_estimators=200,
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    )

    # Fit
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"📊 {target_name} - MAE: {mae:.3f} | R²: {r2:.3f}")

    # ✅ Save model into ../models
    models_dir = "../models"
    os.makedirs(models_dir, exist_ok=True)
    save_path = os.path.join(models_dir, model_filename)
    joblib.dump(model, save_path)
    print(f"💾 Saved model to: {save_path}")


# ============================================
# 4. Train models for each target
# ============================================

if __name__ == "__main__":
    for target, filename in TARGET_COLUMNS.items():
        train_and_save_model(target, filename)

    print("\n✅ All requested models trained and saved.")
