from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import Literal

from para_predictor import predict_para_athlete

app = FastAPI(
    title="Para-Athlete Health API",
    description="Predict stamina, fatigue, and injury risk for para-athletes.",
    version="1.0.0",
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

    training_days_per_week: int = Field(..., ge=1, le=7)
    sleep_hours: float = Field(..., ge=0.0, le=12.0)

    heart_rate_rest: int = Field(..., ge=30, le=120)
    daily_calorie_intake: int = Field(..., ge=800, le=6000)
    protein_intake_g: float = Field(..., ge=0.0, le=400.0)
    water_intake_liters: float = Field(..., ge=0.0, le=10.0)
    hydration_level: int = Field(..., ge=0, le=100)


# ------------------------------------------
# Routes
# ------------------------------------------

@app.get("/")
def root():
    return {
        "message": "Para-Athlete Health API is running.",
        "endpoints": ["/predict"],
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
