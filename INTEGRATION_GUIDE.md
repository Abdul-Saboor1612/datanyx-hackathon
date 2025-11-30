# Backend-Frontend Integration Guide

This guide explains how the backend (datanyx-hackathon) and frontend (shadcn-ui) are integrated.

## Overview

The backend is a FastAPI application that provides:
1. **ML Prediction API** (`/predict`) - Predicts stamina, fatigue, and injury risk
2. **Gemini Coach API** (`/coach/chat`) - Provides AI-powered coaching advice

The frontend is a React/TypeScript application that:
1. Collects athlete data via a form
2. Gets ML predictions
3. Provides a chat interface with the Gemini AI coach

## Architecture

### Backend Workflow
```
User Input → /predict → ML Models → Predictions
                     ↓
              /coach/chat → Gemini AI → Personalized Advice
```

### Frontend Workflow
```
Dashboard (Form) → Get Predictions → Display Results → Chat with AI Coach
```

## Setup Instructions

### Backend Setup

1. Navigate to the backend directory:
   ```bash
   cd para_athlete_project
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Create a `.env` file in `para_athlete_project/`:
   ```env
   GEMINI_API_KEY=your_gemini_api_key_here
   ```

4. Start the FastAPI server:
   ```bash
   cd src
   uvicorn api:app --reload --host 0.0.0.0 --port 8000
   ```

   The API will be available at `http://127.0.0.1:8000`

5. Verify the API is running:
   ```bash
   curl http://127.0.0.1:8000/
   ```

### Frontend Setup

1. Navigate to the frontend directory:
   ```bash
   cd shadcn-ui
   ```

2. Install dependencies:
   ```bash
   pnpm install
   ```

3. Create a `.env` file (or copy from `.env.example`):
   ```bash
   VITE_API_BASE_URL=http://127.0.0.1:8000
   ```

4. Start the development server:
   ```bash
   pnpm dev
   ```

   The frontend will be available at `http://localhost:5173`

## API Endpoints

### POST `/predict`

Predicts stamina, fatigue, and injury risk for a para-athlete.

**Request Body:**
```json
{
  "age": 28,
  "gender": "Male",
  "disability_type": "Spinal Cord Injury",
  "sport_type": "Wheelchair Racing",
  "weight_kg": 75.0,
  "height_cm": 175.0,
  "training_days_per_week": 6,
  "sleep_hours": 7.5,
  "heart_rate_rest": 60,
  "daily_calorie_intake": 2600,
  "protein_intake_g": 130.0,
  "water_intake_liters": 3.0,
  "hydration_level": 75
}
```

**Response:**
```json
{
  "input": { ... },
  "predictions": {
    "stamina_level": 85.5,
    "fatigue_level": 0.25,
    "injury_risk_score": 0.30,
    "injury_risk_label": "Low"
  }
}
```

### POST `/coach/chat`

Get personalized coaching advice from Gemini AI.

**Request Body:**
```json
{
  "athlete_data": { ... },
  "predictions": { ... },
  "conversation_history": "",
  "user_question": "What diet should I follow?"
}
```

**Response:**
```json
{
  "response": "SUMMARY:\n- ...\n\nCALORIES & DIET:\n- ...",
  "question": "What diet should I follow?"
}
```

## Frontend Pages

### Dashboard (`/`)
- Main entry point
- Form to enter athlete data
- Displays predictions after submission
- Button to start chat with AI coach

### Chat (`/chat`)
- Chat interface with Gemini AI coach
- Requires athlete data and predictions from Dashboard
- Maintains conversation history
- Formatted responses with sections (SUMMARY, CALORIES & DIET, etc.)

## Field Mappings

The frontend form fields are mapped to backend fields:

| Frontend Field | Backend Field | Notes |
|---------------|---------------|-------|
| `age` | `age` | Direct mapping (12-70) |
| `gender` | `gender` | Direct mapping (Male/Female/Other) |
| `disabilityType` | `disability_type` | Mapped via `mapDisabilityType()` |
| `sport` | `sport_type` | Mapped via `mapSportType()` |
| `trainingHours` | `training_days_per_week` | Converted via `convertTrainingHoursToDays()` |
| `sleepHours` | `sleep_hours` | Direct mapping (0-12) |
| `weightKg` | `weight_kg` | Direct mapping (30-150) |
| `heightCm` | `height_cm` | Direct mapping (120-210) |
| `heartRateRest` | `heart_rate_rest` | Direct mapping (30-120) |
| `dailyCalorieIntake` | `daily_calorie_intake` | Direct mapping (800-6000) |
| `proteinIntakeG` | `protein_intake_g` | Direct mapping (0-400) |
| `waterIntakeLiters` | `water_intake_liters` | Direct mapping (0-10) |
| `hydrationLevel` | `hydration_level` | Direct mapping (0-100, default 70) |

### Disability Type Mapping

- `lower-limb` → `Amputation`
- `upper-limb` → `Amputation`
- `visual` → `Visual Impairment`
- `cerebral-palsy` → `Cerebral Palsy`
- `spinal-cord` → `Spinal Cord Injury`
- `other` → `Amputation` (fallback)

### Sport Type Mapping

The frontend allows free-text sport input, which is intelligently mapped:
- Contains "wheelchair" + "racing" → `Wheelchair Racing`
- Contains "swimming" → `Para Swimming`
- Contains "powerlifting" → `Para Powerlifting`
- Contains "archery" → `Para Archery`
- Contains "athletics", "track", or "sprint" → `Para Athletics (Track)`
- Default → `Para Athletics (Track)`

## CORS Configuration

The backend includes CORS middleware to allow requests from:
- `http://localhost:5173` (Vite default)
- `http://localhost:3000` (alternative port)
- `http://127.0.0.1:5173`

To add more origins, edit `para_athlete_project/src/api.py`:

```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://your-domain.com"],
    ...
)
```

## Gemini AI Coach

The Gemini coach provides structured responses with:
- **SUMMARY** - Overview of fitness and readiness
- **CALORIES & DIET** - Nutrition recommendations
- **SAMPLE DAY MEAL PLAN** - Personalized meal suggestions
- **ACTIONS (TRAINING + RECOVERY)** - Training and recovery advice

The coach uses:
- Athlete profile data
- ML model predictions
- Sport-specific recommendations
- Conversation history for context

## Testing the Integration

1. Start both backend and frontend servers
2. Navigate to the frontend at `http://localhost:5173`
3. Fill out the athlete form on the Dashboard
4. Submit to get predictions
5. Click "Start Chat with AI Coach" to begin chatting
6. Ask questions about performance, nutrition, or training

## Troubleshooting

### CORS Errors

If you see CORS errors in the browser console:
- Ensure the backend CORS middleware includes your frontend URL
- Check that the backend is running on the expected port
- Verify the `VITE_API_BASE_URL` in the frontend `.env` file

### API Connection Errors

If the frontend can't connect to the backend:
- Verify the backend is running: `curl http://127.0.0.1:8000/`
- Check the `VITE_API_BASE_URL` environment variable
- Ensure no firewall is blocking the connection

### Gemini API Errors

If you get errors from the Gemini coach:
- Verify `GEMINI_API_KEY` is set in the backend `.env` file
- Check that the API key is valid
- Ensure you have quota/access to the Gemini API

### Validation Errors

If you get validation errors:
- Check that all required fields are filled
- Verify numeric fields are within valid ranges (see form validation)
- Ensure disability type and sport are correctly formatted

## File Structure

### Backend
```
para_athlete_project/
├── src/
│   ├── api.py              # FastAPI app with /predict and /coach/chat
│   ├── para_predictor.py   # ML prediction logic
│   └── gemini_coach.py    # CLI reference implementation
├── models/                # Trained ML models
├── requirements.txt
└── .env                    # GEMINI_API_KEY
```

### Frontend
```
shadcn-ui/
├── src/
│   ├── lib/
│   │   ├── api.ts          # API service layer
│   │   └── predictionUtils.ts  # Prediction conversion utilities
│   ├── components/
│   │   ├── AthleteForm.tsx # Form with all backend fields
│   │   ├── PredictionSummary.tsx
│   │   └── ...
│   ├── pages/
│   │   ├── Dashboard.tsx   # Main form and predictions
│   │   └── Chat.tsx        # Gemini coach chat interface
│   └── types/
│       └── index.ts        # TypeScript types
└── .env                    # VITE_API_BASE_URL
```

## Next Steps

- [ ] Add authentication
- [ ] Implement athlete data persistence (database)
- [ ] Add real-time updates
- [ ] Add more prediction endpoints
- [ ] Deploy both applications
- [ ] Add error monitoring and logging
