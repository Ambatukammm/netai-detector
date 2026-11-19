"""
NetAI Anomaly Detector — Inference API
Auto-loads model + scaler — ready to use
"""
from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import torch
import joblib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from netai_detector.models.autoencoder import AnomalyAutoencoder

# ─── AUTO-LOAD EVERYTHING ───
PROJECT_ROOT = Path(__file__).parent.parent
MODEL_INFO_PATH = PROJECT_ROOT / "models" / "model_info.json"
MODEL_PATH = PROJECT_ROOT / "models" / "anomaly_model.pt"
SCALER_PATH = PROJECT_ROOT / "models" / "scalers" / "anomaly_scaler.joblib"

print("[INFO] Loading model info...")
with open(MODEL_INFO_PATH) as f:
    model_info = json.load(f)

THRESHOLD = model_info["threshold"]
INPUT_DIM = model_info["input_dim"]
LATENT_DIM = model_info["latent_dim"]
HIDDEN_DIM = model_info["hidden_dim"]

print("[INFO] Loading model...")
checkpoint = torch.load(MODEL_PATH, weights_only=True, map_location="cpu")
model = AnomalyAutoencoder(
    input_dim=INPUT_DIM,
    latent_dim=LATENT_DIM,
    hidden_dim=HIDDEN_DIM
)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

print("[INFO] Loading scaler...")
scaler_data = joblib.load(SCALER_PATH)
scaler = scaler_data["scaler"]

print(f"✅ API READY — Threshold: {THRESHOLD:.6f}")

# ─── API SETUP ───
app = FastAPI(title="NetAI Anomaly Detector", version="1.0.0")

class FeaturesRequest(BaseModel):
    features: list[float]

# ─── ENDPOINTS ───
@app.get("/")
def root():
    return {
        "message": "NetAI Anomaly Detector API",
        "docs": "/docs",
        "threshold": THRESHOLD
    }

@app.get("/health")
def health():
    return {
        "status": "online",
        "threshold": THRESHOLD,
        "input_dim": INPUT_DIM
    }

@app.post("/detect")
def detect(request: FeaturesRequest):
    x = np.array([request.features], dtype=np.float32)
    x_scaled = scaler.transform(x)
    
    x_tensor = torch.tensor(x_scaled)
    with torch.no_grad():
        reconstruction = model(x_tensor).numpy()
    
    mse = np.mean(np.power(x_scaled - reconstruction, 2), axis=1)[0]
    is_anomaly = bool(mse > THRESHOLD)
    
    return {
        "anomaly": is_anomaly,
        "mse_loss": round(float(mse), 6),
        "threshold": THRESHOLD,
        "status": "ANOMALY" if is_anomaly else "NORMAL"
    }

# ─── RUN ───
if __name__ == "__main__":
    import uvicorn
    print(f"\n🌐 Server: http://localhost:8000")
    print(f"📖 Docs:   http://localhost:8000/docs\n")
    uvicorn.run(app, host="0.0.0.0", port=8000)
