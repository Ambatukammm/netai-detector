"""
NetAI Anomaly Detector — Inference API
FastAPI server for real-time anomaly detection
"""
from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import torch
from pathlib import Path
import sys

# Add src folder so we can import our modules
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from netai_detector.models.autoencoder import AnomalyAutoencoder
import joblib

# ─── CONFIG ───
THRESHOLD = 1.392172
MODEL_PATH = Path(__file__).parent.parent.parent / "models" / "anomaly_model.pt"
SCALER_PATH = Path(__file__).parent.parent.parent / "models" / "scalers" / "anomaly_scaler.joblib"

# ─── LOAD MODEL & SCALER ───
print("[INFO] Loading model...")
model = AnomalyAutoencoder(input_dim=80, latent_dim=10, hidden_dim=40)
model.load_state_dict(torch.load(MODEL_PATH, weights_only=True, map_location="cpu"))
model.eval()

print("[INFO] Loading scaler...")
scaler_data = joblib.load(SCALER_PATH)
scaler = scaler_data["scaler"]

print("[INFO] API Ready ✅")

# ─── API SETUP ───
app = FastAPI(title="NetAI Anomaly Detector", version="1.0.0")

class FeaturesRequest(BaseModel):
    features: list[float]

# ─── ENDPOINTS ───
@app.get("/")
def root():
    return {"message": "NetAI Anomaly Detector API", "docs": "/docs"}

@app.get("/health")
def health():
    return {"status": "online", "threshold": THRESHOLD}

@app.post("/detect")
def detect(request: FeaturesRequest):
    # Convert input to numpy array
    x = np.array([request.features], dtype=np.float32)
    
    # Apply same scaling as training
    x_scaled = scaler.transform(x)
    
    # Run through model
    x_tensor = torch.tensor(x_scaled)
    with torch.no_grad():
        reconstruction = model(x_tensor).numpy()
    
    # Calculate error
    mse = np.mean(np.power(x_scaled - reconstruction, 2), axis=1)[0]
    
    # Decision
    is_anomaly = bool(mse > THRESHOLD)
    
    return {
        "anomaly": is_anomaly,
        "mse_loss": round(float(mse), 6),
        "threshold": THRESHOLD,
        "status": "ANOMALY" if is_anomaly else "NORMAL"
    }

# ─── RUN SERVER ───
if __name__ == "__main__":
    import uvicorn
    print(f"\n🌐 Server running at: http://localhost:8000")
    print(f"📖 API Docs: http://localhost:8000/docs\n")
    uvicorn.run(app, host="0.0.0.0", port=8000)
