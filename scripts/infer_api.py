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
