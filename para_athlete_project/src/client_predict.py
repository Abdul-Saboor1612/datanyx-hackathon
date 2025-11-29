import requests

API_URL = "http://127.0.0.1:8000/predict"  # change to deployed URL later

def call_para_api(payload: dict) -> dict:
    """
    Send athlete data to FastAPI and return predictions.
    """
    resp = requests.post(API_URL, json=payload)
    resp.raise_for_status()
    return resp.json()


if __name__ == "__main__":
    sample_payload = {
        "age": 20,
        "gender": "Male",
        "disability_type": "Amputation",
        "sport_type": "Wheelchair Racing",
        "training_days_per_week": 4,
        "sleep_hours": 7.0,
        "heart_rate_rest": 66,
        "daily_calorie_intake": 2500,
        "protein_intake_g": 60.0,
        "water_intake_liters": 2.2,
        "hydration_level": 65
    }

    result = call_para_api(sample_payload)
    print(result)
