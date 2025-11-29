import os
import re
import difflib
import requests
from google import genai
from dotenv import load_dotenv

# ==========================
#  Load .env + config
# ==========================

load_dotenv()  # expects GEMINI_API_KEY in .env

API_URL = "http://127.0.0.1:8000/predict"

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY environment variable not set")

client = genai.Client(api_key=GEMINI_API_KEY)


# ==========================
#  Helper: call FastAPI
# ==========================

def call_para_api(payload: dict) -> dict:
    resp = requests.post(API_URL, json=payload)
    if not resp.ok:
        print("❌ Backend error:", resp.status_code)
        print("Details:", resp.text)
        resp.raise_for_status()
    return resp.json()


# ==========================
#  CLI input helpers
# ==========================

def ask_int(prompt: str) -> int:
    while True:
        try:
            return int(input(prompt))
        except ValueError:
            print("❌ Please enter a valid integer.")


def ask_float(prompt: str) -> float:
    while True:
        try:
            return float(input(prompt))
        except ValueError:
            print("❌ Please enter a valid number.")


def ask_int_in_range(prompt: str, min_value: int, max_value: int) -> int:
    """Ask for an integer and ensure it's within a valid range."""
    while True:
        value = ask_int(f"{prompt} ({min_value}–{max_value}): ")
        if min_value <= value <= max_value:
            return value
        print(f"❌ Please enter a value between {min_value} and {max_value}.")


def ask_float_in_range(prompt: str, min_value: float, max_value: float) -> float:
    """Ask for a float and ensure it's within a valid range."""
    while True:
        value = ask_float(f"{prompt} ({min_value}–{max_value}): ")
        if min_value <= value <= max_value:
            return value
        print(f"❌ Please enter a value between {min_value} and {max_value}.")


def _normalize(s: str) -> str:
    s = s.lower()
    s = re.sub(r"[^a-z\s]", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def ask_choice(prompt, choices):
    choices_str = "/".join(choices)
    normalized_map = {_normalize(c): c for c in choices}
    normalized_choices = list(normalized_map.keys())

    while True:
        val = input(f"{prompt} ({choices_str}): ").strip()
        if not val:
            print("❌ Please enter something.")
            continue

        norm_val = _normalize(val)

        # exact-ish
        for c in choices:
            if norm_val == _normalize(c):
                return c

        # fuzzy
        close = difflib.get_close_matches(norm_val, normalized_choices, n=1, cutoff=0.6)
        if close:
            best_norm = close[0]
            best_choice = normalized_map[best_norm]
            confirm = input(f"Did you mean '{best_choice}'? (y/n): ").strip().lower()
            if confirm in ("y", "yes"):
                return best_choice

        print(f"❌ Could not understand '{val}'. Please choose one of: {choices_str}")


# ==========================
#  Main CLI with chat loop
# ==========================

def main():
    print("=== Para-Athlete Gemini Coach CLI ===\n")

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

    # 1) Collect athlete data
    print("Enter athlete details:\n")
    age = ask_int_in_range("Age (years)", 12, 70)
    gender = ask_choice("Gender", gender_choices)
    disability_type = ask_choice("Disability type", disability_choices)
    sport_type = ask_choice("Sport type", sport_choices)

    weight_kg = ask_float_in_range("Weight (kg)", 30.0, 150.0)
    height_cm = ask_float_in_range("Height (cm)", 120.0, 210.0)

    training_days_per_week = ask_int_in_range("Training days per week", 1, 7)
    sleep_hours = ask_float_in_range("Average sleep hours per night", 0.0, 12.0)

    heart_rate_rest = ask_int_in_range("Resting heart rate (bpm)", 30, 120)
    daily_calorie_intake = ask_int_in_range("Daily calorie intake (kcal)", 800, 6000)
    protein_intake_g = ask_float_in_range("Daily protein intake (grams)", 0.0, 400.0)
    water_intake_liters = ask_float_in_range("Daily water intake (liters)", 0.0, 10.0)

    # simple BMI for Gemini context (not used by ML)
    height_m = height_cm / 100.0 if height_cm > 0 else 0
    bmi = round(weight_kg / (height_m ** 2), 1) if height_m > 0 else None

    athlete_input = {
        "age": age,
        "gender": gender,
        "disability_type": disability_type,
        "sport_type": sport_type,
        "weight_kg": weight_kg,
        "height_cm": height_cm,
        "training_days_per_week": training_days_per_week,
        "sleep_hours": sleep_hours,
        "heart_rate_rest": heart_rate_rest,
        "daily_calorie_intake": daily_calorie_intake,
        "protein_intake_g": protein_intake_g,
        "water_intake_liters": water_intake_liters,
        # hydration_level omitted → API default (70) will be used
    }

    print("\n📡 Calling ML API for predictions...")
    api_result = call_para_api(athlete_input)
    preds = api_result["predictions"]
    print("✅ Got predictions from backend.\n")

    conversation_history = ""

    # SYSTEM PROMPT: sport-specific, personalized diet plan
    system_prompt = """
You are a para-athlete performance and nutrition coach AI.

Inputs you receive:
- athlete_profile: age, gender, disability_type, sport_type, weight_kg, height_cm,
  training_days_per_week, sleep_hours.
- nutrition: daily_calorie_intake, protein_intake_g, water_intake_liters.
- derived_metrics: BMI (approx).
- model predictions: stamina_level (0–100), fatigue_level (0–10),
  injury_risk_score (0–1), injury_risk_label.

Rules:
- ALWAYS base your answers on the given athlete_profile, sport_type, and predictions.
- If asked about tournament chances, do NOT say you can't answer.
  Instead, explain their current readiness, strengths, and gaps, and what to improve.
- For nutrition:
  - Suggest a realistic daily calorie RANGE based on sport_type, weight_kg, height_cm
    and training_days_per_week.
  - State clearly if current intake is below / within / above that range.
  - Always give a personalized diet plan tailored to their sport:
      * Endurance (Wheelchair Racing, Para Swimming, Para Athletics): higher carbs.
      * Strength/power (Para Powerlifting): higher protein and adequate carbs.
      * Precision / stability (Para Archery): balanced energy, steady blood sugar.
- Provide a simple SAMPLE DAY MEAL PLAN with meal names (Breakfast, Lunch, Snack, Dinner)
  and example foods with approximate ideas (not exact grams).

RESPONSE FORMAT (STRICT):

SUMMARY (max 3 bullets):
- ...

CALORIES & DIET (max 3 bullets):
- ...

SAMPLE DAY MEAL PLAN (max 6 bullets):
- Breakfast: ...
- Mid-morning snack: ...
- Lunch: ...
- Pre-training snack: ...
- Post-training: ...
- Dinner: ...

ACTIONS (TRAINING + RECOVERY) (max 4 bullets):
- ...

Do NOT add any other sections, disclaimers or long paragraphs.
"""

    print("You can now chat with the coach. Type 'exit' to stop.\n")

    while True:
        user_question = input("You: ").strip()
        if user_question.lower() in ("exit", "quit", "q"):
            print("👋 Ending chat. Take care!")
            break

        if not user_question:
            user_question = "Give me an overview of my fitness, risk and what diet I should follow."

        conversation_text = f"""
Athlete data:
{athlete_input}

Derived metrics:
BMI: {bmi}

Model predictions:
{preds}

Previous conversation:
{conversation_history}

Current user question:
{user_question}
"""

        print("\n🤖 Coach is thinking...\n")

        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[
                system_prompt,
                conversation_text,
            ],
        )

        coach_reply = response.text.strip()
        print("===== GEMINI COACH (FILTERED) =====\n")
        print(coach_reply)
        print("\n===================================\n")

        conversation_history += f"\nUser: {user_question}\nCoach: {coach_reply}\n"


if __name__ == "__main__":
    main()
