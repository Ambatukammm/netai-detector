"""
NetAI Anomaly Detector — Auto-Training
Captures live traffic → trains autoencoder → saves everything automatically
"""
import sys
import yaml
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
import json

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from netai_detector.models.autoencoder import AnomalyAutoencoder
from netai_detector.features.scaler import FeatureScaler

# ─── PATHS ───
PROJECT_ROOT = Path(__file__).parent.parent
DATA_FILE = PROJECT_ROOT / "data" / "normal_traffic.npy"
MODEL_DIR = PROJECT_ROOT / "models"
SCALER_DIR = MODEL_DIR / "scalers"

MODEL_DIR.mkdir(exist_ok=True)
SCALER_DIR.mkdir(exist_ok=True)

# ─── LOAD OR CAPTURE DATA ───
print("="*60)
print("🚀 NETAI ANOMALY DETECTOR — TRAINING")
print("="*60)

if DATA_FILE.exists():
    print(f"[INFO] Using existing data: {DATA_FILE}")
    X_train = np.load(DATA_FILE)
else:
    print("[INFO] No traffic data found — starting live capture...")
    import sys
sys.path.insert(0, str(Path(__file__).parent))
from live_capture import capture_training_data
    X_train = capture_training_data(duration=300)  # 5 minutes
    
    if X_train is None or len(X_train) < 10:
        print("[ERROR] Not enough traffic data captured. Try again.")
        sys.exit(1)

# Split train/validation
np.random.shuffle(X_train)
split = int(0.85 * len(X_train))
X_val = X_train[split:]
X_train = X_train[:split]

print(f"\n[DATA] Training: {len(X_train)} windows | Validation: {len(X_val)} windows")
print(f"   Features per window: {X_train.shape[1]}")

# ─── SCALE FEATURES ───
print("\n[INFO] Fitting scaler...")
scaler = FeatureScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

# Save scaler
scaler_path = SCALER_DIR / "anomaly_scaler.joblib"
scaler.save(scaler_path)
print(f"[SAVED] Scaler → {scaler_path}")

# ─── PREPARE TENSORS ───
X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
X_val_tensor = torch.tensor(X_val_scaled, dtype=torch.float32)

# ─── MODEL SETUP ───
input_dim = X_train.shape[1]
latent_dim = 16
hidden_dim = 64

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Training on: {device}")

model = AnomalyAutoencoder(
    input_dim=input_dim,
    latent_dim=latent_dim,
    hidden_dim=hidden_dim
).to(device)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# ─── TRAINING LOOP ───
batch_size = 32
epochs = 50

print(f"\n[INFO] Starting training: {epochs} epochs")
model.train()

for epoch in range(epochs):
    perm = torch.randperm(len(X_train_tensor))
    train_loss_sum = 0
    
    for i in range(0, len(X_train_tensor), batch_size):
        idx = perm[i:i+batch_size]
        xb = X_train_tensor[idx].to(device)
        
        optimizer.zero_grad()
        output = model(xb)
        loss = criterion(output, xb)
        loss.backward()
        optimizer.step()
        train_loss_sum += loss.item()
    
    # Validation
    model.eval()
    with torch.no_grad():
        val_out = model(X_val_tensor.to(device))
        val_loss = criterion(val_out, X_val_tensor.to(device)).item()
    model.train()
    
    print(f"  Epoch {epoch+1:02d} | Train Loss: {train_loss_sum/len(X_train_tensor):.6f} | Val Loss: {val_loss:.6f}")

# ─── SAVE MODEL ───
model_path = MODEL_DIR / "anomaly_model.pt"
torch.save({
    'model_state_dict': model.state_dict(),
    'input_dim': input_dim,
    'latent_dim': latent_dim,
    'hidden_dim': hidden_dim
}, model_path)
print(f"\n[SAVED] Model → {model_path}")

# ─── CALCULATE THRESHOLD ───
model.eval()
with torch.no_grad():
    reconstructions = model(X_val_tensor.to(device)).cpu().numpy()
    mse = np.mean(np.power(X_val_scaled - reconstructions, 2), axis=1)

threshold = np.percentile(mse, 99.9)

# Save all config for API
model_info = {
    "threshold": float(threshold),
    "input_dim": input_dim,
    "latent_dim": latent_dim,
    "hidden_dim": hidden_dim,
    "trained_on": time.strftime("%Y-%m-%d %H:%M:%S")
}

info_path = MODEL_DIR / "model_info.json"
with open(info_path, "w") as f:
    json.dump(model_info, f, indent=2)

print(f"\n{'='*60}")
print(f"✅ TRAINING COMPLETE")
print(f"   Threshold (99.9th percentile): {threshold:.6f}")
print(f"   Values ABOVE {threshold:.6f} = ANOMALY 🚨")
print(f"{'='*60}")
