import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Literal, Optional
from dotenv import load_dotenv
from google import genai

from para_predictor import predict_para_athlete

# Load environment variables
load_dotenv()

# Initialize Gemini client
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
gemini_client = None
if GEMINI_API_KEY:
    try:
        gemini_client = genai.Client(api_key=GEMINI_API_KEY)
    except Exception as e:
        print(f"Warning: Could not initialize Gemini client: {e}")

app = FastAPI(
    title="Para-Athlete Health API",
    description="Predict stamina, fatigue, and injury risk for para-athletes.",
    version="1.0.0",
)

# Add CORS middleware to allow frontend requests
# For development: Allow all localhost origins (any port) using regex
# This allows any port on localhost/127.0.0.1 (e.g., 5173, 5174, 3000, etc.)
app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r"https?://(localhost|127\.0\.0\.1|0\.0\.0\.0)(:\d+)?",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------------------------------
# Request body schema
# ------------------------------------------

class ParaAthleteInput(BaseModel):
    age: int = Field(..., ge=12, le=70)

    gender: Literal["Male", "Female", "Other"]
    disability_type: Literal[
        "Amputation",
        "Visual Impairment",
        "Cerebral Palsy",
        "Spinal Cord Injury",
        "Intellectual Impairment",
    ]
    sport_type: Literal[
        "Wheelchair Racing",
        "Para Swimming",
        "Para Powerlifting",
        "Para Athletics (Track)",
        "Para Archery",
    ]

    # ✅ New: body size info for nutrition logic
    weight_kg: float = Field(..., ge=30.0, le=150.0)
    height_cm: float = Field(..., ge=120.0, le=210.0)

    training_days_per_week: int = Field(..., ge=1, le=7)
    sleep_hours: float = Field(..., ge=0.0, le=12.0)

    heart_rate_rest: int = Field(..., ge=30, le=120)
    daily_calorie_intake: int = Field(..., ge=800, le=6000)
    protein_intake_g: float = Field(..., ge=0.0, le=400.0)
    water_intake_liters: float = Field(..., ge=0.0, le=10.0)

    # Optional: we default this now
    hydration_level: int = Field(70, ge=0, le=100)


# ------------------------------------------
# Routes
# ------------------------------------------

@app.get("/")
def root():
    return {
        "message": "Para-Athlete Health API is running.",
        "endpoints": ["/predict", "/coach/chat"],
        "gemini_enabled": gemini_client is not None,
    }


@app.post("/predict")
def predict(input_data: ParaAthleteInput):
    """
    Predict stamina_level, fatigue_level, and injury_risk_score
    for a para-athlete profile.
    """
    features = input_data.dict()
    preds = predict_para_athlete(features)

    return {
        "input": features,
        "predictions": preds,
    }


# ------------------------------------------
# Gemini Coach endpoints
# ------------------------------------------

class CoachChatRequest(BaseModel):
    athlete_data: ParaAthleteInput
    predictions: dict
    conversation_history: Optional[str] = ""
    user_question: str


@app.post("/coach/chat")
def coach_chat(request: CoachChatRequest):
    """
    Get personalized coaching advice from Gemini AI based on athlete data and predictions.
    """
    if not gemini_client:
        raise HTTPException(
            status_code=503,
            detail="Gemini AI service is not available. Please set GEMINI_API_KEY environment variable."
        )

    athlete_input = request.athlete_data.dict()
    preds = request.predictions

    # Calculate BMI for context
    height_m = athlete_input["height_cm"] / 100.0 if athlete_input["height_cm"] > 0 else 0
    bmi = round(athlete_input["weight_kg"] / (height_m ** 2), 1) if height_m > 0 else None

    # System prompt - focused and contextual responses
    system_prompt = """
You are a para-athlete performance and nutrition coach AI. Be concise and focused.

Inputs you receive:
- athlete_profile: age, gender, disability_type, sport_type, weight_kg, height_cm,
  training_days_per_week, sleep_hours.
- nutrition: daily_calorie_intake, protein_intake_g, water_intake_liters.
- derived_metrics: BMI (approx).
- model predictions: stamina_level (0–100), fatigue_level (0–10),
  injury_risk_score (0–1), injury_risk_label.

IMPORTANT RULES:
1. ANSWER ONLY WHAT IS ASKED. If they ask about diet, focus on diet. If they ask about training, focus on training.
2. Do NOT provide the full format unless specifically asked for a complete overview.
3. Be conversational and direct - no unnecessary sections.
4. If asked a specific question, give a focused answer (2-4 bullet points max).
5. Only provide the full format (SUMMARY, CALORIES & DIET, etc.) if they ask for "overview", "complete analysis", or "full report".

For diet questions:
- Focus on what they asked (e.g., "how to improve diet" = specific improvements, not full meal plan)
- Only provide meal plan if explicitly requested
- Give actionable, specific advice

For training questions:
- Focus on training-specific advice
- Reference their current predictions only if relevant

For general questions:
- Keep it brief and actionable
- Reference predictions only when relevant to the answer

Example good responses:
- "How to improve diet?" → "Increase calories to 2500-3200 kcal/day. Add 30-50g more protein. Focus on complex carbs before training."
- "What about my stamina?" → "Your stamina is low (38%). Increase carb intake, especially before training. Ensure 7-9 hours sleep."
"""

    # Default question if empty
    user_question = request.user_question.strip()
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
{request.conversation_history}

Current user question:
{user_question}
"""

    try:
        response = gemini_client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[
                system_prompt,
                conversation_text,
            ],
        )

        coach_reply = response.text.strip()
        return {
            "response": coach_reply,
            "question": user_question,
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error calling Gemini API: {str(e)}"
        )
