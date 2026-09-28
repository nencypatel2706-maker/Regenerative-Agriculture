import os
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from services.ai_engine import ArginAIEngine

app = FastAPI(title="AgriN - Regenerative Agricultural Intelligence API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

try:
    ai_engine = ArginAIEngine()
except Exception as exc:
    ai_engine = None
    startup_error = str(exc)

class AdvisoryRequest(BaseModel):
    soil_ph: float | None = None
    carbon_level: float | None = None
    rainfall: float | None = None
    forecast: str = "Normal monsoon"
    crop: str = "Cotton"
    region: str = "Gujarat"

class ChatRequest(BaseModel):
    question: str
    crop: str = "Cotton"
    region: str = "Gujarat"

@app.get("/api/health")
async def health():
    return {"status": "ok", "gemini_configured": ai_engine is not None}

@app.post("/api/diagnose")
async def diagnose_crop(file: UploadFile = File(...), region: str = Form("Gujarat")):
    if ai_engine is None:
        raise HTTPException(503, startup_error)
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(400, "Please upload an image file.")
    image_bytes = await file.read()
    analysis = ai_engine.diagnose_crop_disease(image_bytes, file.content_type, region)
    return {"status": "success", "advisory": analysis}

@app.post("/api/advisory")
async def get_advisory(req: AdvisoryRequest):
    if ai_engine is None:
        raise HTTPException(503, startup_error)
    soil_data = {"ph": req.soil_ph, "carbon_level": req.carbon_level}
    weather_data = {"forecast": req.forecast, "rainfall": req.rainfall}
    plan = ai_engine.generate_regenerative_advisory(soil_data, weather_data, req.crop, req.region)
    return {"status": "success", "regenerative_plan": plan}

@app.post("/api/chat")
async def chat(req: ChatRequest):
    if ai_engine is None:
        raise HTTPException(503, startup_error)
    prompt = f"""You are AgriN, an agricultural assistant for Indian farmers.
Crop: {req.crop}
Region: {req.region}
Farmer question: {req.question}
Answer simply and practically. Support regenerative agriculture where relevant. Do not invent weather/soil data."""
    response = ai_engine.client.models.generate_content(model=ai_engine.model_id, contents=prompt)
    return {"status": "success", "answer": response.text or "No answer returned."}

app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")
