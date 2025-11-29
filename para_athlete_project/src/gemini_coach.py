import re
import difflib
import requests
import google.generativeai as genai
from PIL import Image
import io
import os

# ==========================
#  CONFIG
# ==========================

API_URL = "http://127.0.0.1:8000/predict"  # FastAPI endpoint
GEMINI_API_KEY = "AIzaSyBQCAcMNBuDXIUZdXhHKSUn3fG7cWobEyQ"  # Your API key

# Configure the API
genai.configure(api_key=GEMINI_API_KEY)

# Initialize the model
model = genai.GenerativeModel('gemini-2.5-flash')

# ==========================
#  Helper: call FastAPI
# ==========================

def call_para_api(payload: dict) -> dict:
    resp = requests.post(API_URL, json=payload)
    resp.raise_for_status()
    return resp.json()


# ==========================
#  CLI input helpers
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


def _similar(a, b):
    """Calculate similarity ratio between two strings (0-1)"""
    return difflib.SequenceMatcher(None, a, b).ratio()


def ask_choice(prompt, choices):
    choices_lower = [c.lower() for c in choices]
    choices_str = "/".join(choices)
    
    while True:
        val = input(f"{prompt} ({choices_str}): ").strip()
        
        # Remove extra spaces and normalize case
        val = re.sub(r'\s+', ' ', val).strip().lower()
        
        # Check for direct match
        if val in choices_lower:
            return choices[choices_lower.index(val)]
            
        # Check for partial matches (e.g., 'vis' matches 'Visual Impairment')
        matches = [c for c in choices if val in c.lower()]
        if len(matches) == 1:
            return matches[0]
            
        # Check for similar words with small typos
        for choice in choices:
            # If input is at least 3 characters and similar to a choice
            if len(val) > 2 and _similar(val, choice.lower()) > 0.7:
                return choice
                
        print(f"❌ Please choose one of: {choices_str}")


# ==========================
#  Main CLI
# ==========================

def analyze_injury(image, athlete_data):
    """Analyze injury image using Gemini Vision"""
    # Convert image to bytes
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format=image.format)
    img_byte_arr = img_byte_arr.getvalue()
    
    prompt = f"""
    You are a sports medicine specialist. Analyze this injury image for a para-athlete.
    
    Athlete details:
    - Age: {athlete_data.get('age')}
    - Gender: {athlete_data.get('gender')}
    - Disability: {athlete_data.get('disability_type')}
    - Sport: {athlete_data.get('sport_type')}
    
    Please provide:
    1. Description of visible injury
    2. Severity assessment (mild/moderate/severe)
    3. Recommended immediate actions
    4. Whether medical attention is advised
    """
    
    response = model.generate_content([prompt, img_byte_arr])
    return response.text

def ask_for_image(prompt):
    """Prompt user to upload an image and validate it"""
    while True:
        try:
            image_path = input(f"\n{prompt} (or press Enter to skip): ").strip()
            if not image_path:  # User pressed Enter
                return None
                
            if not os.path.exists(image_path):
                print("❌ File not found. Please enter a valid file path.")
                continue
                
            try:
                img = Image.open(image_path)
                return img
            except Exception as e:
                print(f"❌ Error loading image: {e}")
                continue
                
        except KeyboardInterrupt:
            print("\nOperation cancelled.")
            return None

def main():
    print("=== Para-Athlete Gemini Coach ===\n")
    print("📷 You can upload injury photos for analysis when prompted.\n")

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

    # Check if user wants to upload an injury image
    injury_img = ask_for_image("Would you like to upload an injury photo for analysis?")
    
    if injury_img:
        print("\n🔍 Analyzing injury image...")
        injury_analysis = analyze_injury(injury_img, athlete_input)
        model_input += f"\n\nInjury Analysis:\n{injury_analysis}"
    
    # Combine system prompt and user input into a single message
    full_prompt = f"{system_prompt}\n\n{model_input}"
    
    # Generate response
    response = model.generate_content(full_prompt)

    print("===== GEMINI COACH RESPONSE =====\n")
    print(response.text)
    print("\n=================================\n")


if __name__ == "__main__":
    main()