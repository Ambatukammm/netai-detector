"""
NetAI Detector — FULL AUTO
Capture traffic → Train model → Start API — ONE CLICK
"""
import sys
import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent

def run_step(name, cmd):
    print(f"\n{'='*60}")
    print(f"▶️  {name}")
    print(f"{'='*60}")
    result = subprocess.run(cmd, cwd=PROJECT_ROOT, shell=True)
    if result.returncode != 0:
        print(f"\n❌ FAILED at: {name}")
        sys.exit(1)
    print(f"✅ {name} — COMPLETE")

if __name__ == "__main__":
    print("🚀 NETAI DETECTOR — FULL AUTO MODE")
    print("==========================================")
    print("1. Capture normal traffic (5 min)")
    print("2. Train anomaly detector")
    print("3. Start detection API")
    print("==========================================\n")
    
    run_step("CAPTURE TRAFFIC", "python scripts/live_capture.py train")
    run_step("TRAIN MODEL", "python scripts/train_anomaly.py")
    
    print(f"\n{'='*60}")
    print("🌐 STARTING DETECTION API")
    print(f"   Local:   http://localhost:8000")
    print(f"   Docs:    http://localhost:8000/docs")
    print(f"{'='*60}\n")
    subprocess.run("python scripts/infer_api.py", cwd=PROJECT_ROOT, shell=True)
