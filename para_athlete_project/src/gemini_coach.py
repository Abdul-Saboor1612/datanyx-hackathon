import os
import re
import difflib
import requests
from google import genai

from dotenv import load_dotenv

# Load .env from current folder
load_dotenv()

# ==========================
#  CONFIG
# ==========================

API_URL = "http://127.0.0.1:8000/predict"  # FastAPI endpoint

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
#  Small helpers
# ==========================

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


def _normalize(s: str) -> str:
    """
    Normalize a string for fuzzy matching:
    - lowercase
    - remove non-letters/spaces
    - collapse multiple spaces
    """
    s = s.lower()
    s = re.sub(r"[^a-z\s]", "", s)  # keep only letters + spaces
    s = re.sub(r"\s+", " ", s).strip()
    return s


def ask_choice(prompt, choices):
    """
    Ask user to choose from a list of choices, but:
    - ignore case
    - handle small typos using fuzzy matching
    """
    choices_str = "/".join(choices)
    # Precompute normalized versions of each choice
    normalized_map = { _normalize(c): c for c in choices }
    normalized_choices = list(normalized_map.keys())

    while True:
        val = input(f"{prompt} ({choices_str}): ").strip()
        if not val:
            print("❌ Please enter something.")
            continue

        norm_val = _normalize(val)

        # 1) Exact case-insensitive match
        for c in choices:
            if norm_val == _normalize(c):
                return c

        # 2) Fuzzy match using difflib
        close = difflib.get_close_matches(norm_val, normalized_choices, n=1, cutoff=0.6)
        if close:
            best_norm = close[0]
            best_choice = normalized_map[best_norm]
            confirm = input(f"Did you mean '{best_choice}'? (y/n): ").strip().lower()
            if confirm in ("y", "yes"):
                return best_choice

        print(f"❌ Could not understand '{val}'. Please choose one of: {choices_str}")


# ==========================
#  Main CLI
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
    age = ask_int("Age (years): ")
    gender = ask_choice("Gender", gender_choices)
    disability_type = ask_choice("Disability type", disability_choices)
    sport_type = ask_choice("Sport type", sport_choices)

    training_days_per_week = ask_int("Training days per week (1–7): ")
    sleep_hours = ask_float("Average sleep hours per night: ")

    heart_rate_rest = ask_int("Resting heart rate (bpm): ")
    daily_calorie_intake = ask_int("Daily calorie intake (kcal): ")
    protein_intake_g = ask_float("Daily protein intake (grams): ")
    water_intake_liters = ask_float("Daily water intake (liters): ")
    hydration_level = ask_int("Hydration level (0–100): ")

    athlete_input = {
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

    print("\n📡 Calling ML API for predictions...")
    api_result = call_para_api(athlete_input)
    preds = api_result["predictions"]
    print("✅ Got predictions from backend.\n")

    # Optional: let user ask something
    user_question = input(
        "What do you want to ask the coach? (e.g., 'How is my fitness and risk?')\n> "
    )
    if not user_question.strip():
        user_question = "Give me an overview of my fitness, fatigue and injury risk and how to improve."

    # 2) Build prompts for Gemini
    system_prompt = """
You are a para-athlete performance coach AI.

You get:
1) The athlete's profile and training data.
2) Model predictions: stamina_level (0–100), fatigue_level (0–10),
   injury_risk_score (0–1), injury_risk_label.

TASK:
- Explain in simple language how their stamina, fatigue, and injury risk look.
- Mention specific numbers briefly but focus more on interpretation.
- Give 3–5 practical suggestions (training, sleep, hydration, nutrition).
- Be supportive, encouraging, and not alarming.
"""

    model_input = f"""
Athlete data:
{athlete_input}

Model predictions:
{preds}

User question:
{user_question}
"""

    print("🤖 Asking Gemini to act as a coach...\n")

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=[
            system_prompt,
            model_input
        ]
    )

    print("===== GEMINI COACH RESPONSE =====\n")
    print(response.text)
    print("\n=================================\n")


if __name__ == "__main__":
    main()
