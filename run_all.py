"""
NetAI Detector — One-Click Master Script
Runs: Train → Export ONNX → Start API
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
    print("🚀 NETAI DETECTOR — ONE-CLICK LAUNCH")
    print(f"📂 Project: {PROJECT_ROOT}")
    
    # Step 1: Train
    run_step("TRAINING", "python scripts/train_anomaly.py")
    
    # Step 2: Export to ONNX
    run_step("EXPORTING ONNX", "python scripts/export_onnx.py")
    
    # Step 3: Start API Server
    print(f"\n{'='*60}")
    print("🌐 STARTING API SERVER — Press Ctrl+C to stop")
    print(f"   Local:   http://localhost:8000")
    print(f"   Docs:    http://localhost:8000/docs")
    print(f"{'='*60}\n")
    subprocess.run("python scripts/infer_api.py", cwd=PROJECT_ROOT, shell=True)
