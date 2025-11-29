import numpy as np
import pandas as pd

# ----------------------------------------------------
# Synthetic Para-Athlete Dataset Generator
# ----------------------------------------------------

def generate_para_athlete_data(n_samples: int = 1500, random_state: int = 42) -> pd.DataFrame:
    np.random.seed(random_state)

    # ---- 1. Categorical pools ----
    genders = ["Male", "Female", "Other"]
    disability_types = [
        "Amputation", "Visual Impairment", "Cerebral Palsy",
        "Spinal Cord Injury", "Intellectual Impairment"
    ]
    sport_types = [
        "Wheelchair Racing", "Para Swimming", "Para Powerlifting",
        "Para Athletics (Track)", "Para Archery"
    ]

    # ---- 2. Basic demographics ----
    athlete_id = np.arange(1, n_samples + 1)
    age = np.clip(np.random.normal(loc=27, scale=6, size=n_samples), 16, 50).round().astype(int)
    gender = np.random.choice(genders, size=n_samples, p=[0.6, 0.35, 0.05])
    disability_type = np.random.choice(disability_types, size=n_samples)
    sport_type = np.random.choice(sport_types, size=n_samples)

    # ---- 3. Training load & session metrics ----
    # training hours per day between 0.5 and 4.5
    training_hours_per_day = np.round(np.random.uniform(0.5, 4.5, size=n_samples), 2)

    # RPE (Rate of Perceived Exertion) 1–10
    rpe_score = np.random.randint(3, 10, size=n_samples)

    # training days per week: 3–7
    training_days_per_week = np.random.randint(3, 8, size=n_samples)

    # weekly training load = hours * RPE * days
    weekly_training_load = np.round(
        training_hours_per_day * rpe_score * training_days_per_week,
        1
    )

    # ---- 4. Sleep & recovery ----
    # sleep hours 4–9 with small noise
    sleep_hours = np.clip(np.random.normal(7, 1.2, size=n_samples), 4, 9).round(1)

    # sleep quality 0–10, slightly higher if more sleep
    sleep_quality_base = 5 + (sleep_hours - 7) * 0.8 + np.random.normal(0, 1, size=n_samples)
    sleep_quality_score = np.clip(sleep_quality_base, 0, 10).round(1)

    # ---- 5. Heart rate & physiology ----
    # Fitter athletes (higher training load) → lower resting HR
    fitness_factor = (weekly_training_load - weekly_training_load.mean()) / weekly_training_load.std()
    heart_rate_rest = 60 - 4 * fitness_factor + np.random.normal(0, 5, size=n_samples)
    heart_rate_rest = np.clip(heart_rate_rest, 45, 90).round(0)

    heart_rate_avg = heart_rate_rest + np.random.uniform(30, 60, size=n_samples)
    heart_rate_max = heart_rate_avg + np.random.uniform(10, 30, size=n_samples)

    heart_rate_avg = np.clip(heart_rate_avg, 90, 180).round(0)
    heart_rate_max = np.clip(heart_rate_max, 120, 210).round(0)

    # ---- 6. Nutrition & hydration ----
    # base calorie need using rough heuristic from age & training
    base_cal = 1800 + training_hours_per_day * 250 + np.random.normal(0, 150, size=n_samples)
    daily_calorie_intake = np.clip(base_cal, 1500, 4000).round(0)

    # macros (rough split)
    protein_intake_g = np.clip(
        training_hours_per_day * np.random.uniform(25, 35, size=n_samples) +
        np.random.normal(0, 10, size=n_samples),
        40, 220
    ).round(0)

    carbohydrate_intake_g = np.clip(
        training_hours_per_day * np.random.uniform(60, 90, size=n_samples) +
        np.random.normal(0, 30, size=n_samples),
        100, 600
    ).round(0)

    fat_intake_g = np.clip(
        daily_calorie_intake * np.random.uniform(0.2, 0.3, size=n_samples) / 9,
        30, 150
    ).round(0)

    # hydration level & water intake
    water_intake_liters = np.clip(
        1.5 + training_hours_per_day * np.random.uniform(0.4, 0.7, size=n_samples) +
        np.random.normal(0, 0.3, size=n_samples),
        1.0, 6.0
    ).round(2)

    hydration_level = np.clip(
        60 + (water_intake_liters - 2.5) * 12 +
        np.random.normal(0, 8, size=n_samples),
        30, 100
    ).round(0)

    # ---- 7. Fatigue, soreness, mood, stamina (core targets) ----
    # fatigue higher if high training load & poor sleep
    fatigue_raw = (
        (weekly_training_load - weekly_training_load.min()) /
        (weekly_training_load.max() - weekly_training_load.min()) * 6
        - (sleep_hours - 7) * 0.8
        + np.random.normal(3, 1.5, size=n_samples)
    )
    fatigue_level = np.clip(fatigue_raw, 0, 10).round(1)

    # muscle soreness related to fatigue & RPE
    muscle_soreness_raw = (
        0.6 * fatigue_level +
        0.3 * (rpe_score - 3) +
        np.random.normal(0, 1, size=n_samples)
    )
    muscle_soreness_level = np.clip(muscle_soreness_raw, 0, 10).round(1)

    # mood, motivation, stress
    mood_score = np.clip(
        7 - 0.3 * fatigue_level + 0.2 * sleep_quality_score +
        np.random.normal(0, 1, size=n_samples),
        0, 10
    ).round(1)

    motivation_level = np.clip(
        6 + 0.3 * (weekly_training_load > weekly_training_load.mean()) -
        0.2 * fatigue_level +
        np.random.normal(0, 1, size=n_samples),
        0, 10
    ).round(1)

    stress_level = np.clip(
        3 + 0.4 * fatigue_level - 0.2 * sleep_quality_score +
        np.random.normal(0, 1, size=n_samples),
        0, 10
    ).round(1)

    # stamina level: higher with load & sleep, lower with fatigue
    stamina_raw = (
        40
        + 2.5 * training_hours_per_day
        + 3 * (sleep_hours - 6)
        - 3 * (fatigue_level - 5)
        + np.random.normal(0, 8, size=n_samples)
    )
    stamina_level = np.clip(stamina_raw, 0, 100).round(1)

    # performance score combines stamina, mood, and fatigue
    performance_raw = (
        0.5 * stamina_level +
        3 * mood_score -
        2 * fatigue_level +
        np.random.normal(0, 5, size=n_samples)
    ) / 1.2
    performance_score = np.clip(performance_raw, 0, 100).round(1)

    # ---- 8. Injury risk & overtraining ----
    injury_risk_score_raw = (
        0.08 * fatigue_level +
        0.06 * muscle_soreness_level +
        0.04 * stress_level -
        0.03 * sleep_quality_score +
        np.random.normal(0, 0.05, size=n_samples)
    )
    injury_risk_score = np.clip(injury_risk_score_raw, 0, 1).round(2)

    overtraining_alert = ((fatigue_level >= 7) & (weekly_training_load > np.percentile(weekly_training_load, 70))).astype(int)

    # ---- 9. Build DataFrame ----
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

        "fatigue_level": fatigue_level,

        "stamina_level": stamina_level,          # 🔥 model target candidate
        "performance_score": performance_score,  # 🔥 model target candidate
        "injury_risk_score": injury_risk_score,  # 🔥 model target candidate
        "overtraining_alert": overtraining_alert # 0/1 label
    }

    df = pd.DataFrame(data)
    return df


if __name__ == "__main__":
    # Generate 1500 records and save to CSV
    df_para = generate_para_athlete_data(n_samples=1500, random_state=42)
    print(df_para.head())
    print("\nShape:", df_para.shape)

    df_para.to_csv("para_athlete_synthetic_data.csv", index=False)
    print("\nSaved to para_athlete_synthetic_data.csv")