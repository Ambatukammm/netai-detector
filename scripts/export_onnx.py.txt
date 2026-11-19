"""
Export Trained PyTorch Models → ONNX Format
Run AFTER training completes
"""
import sys
import yaml
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from netai_detector.models.autoencoder import AnomalyAutoencoder
from netai_detector.models.malware_mlp import MalwareMLP


def export_anomaly(cfg, data_cfg):
    print("\n" + "="*50)
    print("EXPORTING: Anomaly Autoencoder")
    print("="*50)

    model_cfg = cfg["anomaly"]
    model = AnomalyAutoencoder(
        input_dim=model_cfg["input_dim"],
        latent_dim=model_cfg["latent_dim"],
        hidden_dim=model_cfg["hidden_dim"]
    )

    pt_path = Path(data_cfg["paths"]["models"]) / "anomaly_model.pt"
    if not pt_path.exists():
        print(f"[SKIP] {pt_path} not found — train first!")
        return

    model.load_state_dict(torch.load(pt_path, weights_only=True))
    model.eval()

    dummy_input = torch.randn(1, model_cfg["input_dim"])
    onnx_path = Path(data_cfg["paths"]["models"]) / "anomaly_model.onnx"

    torch.onnx.export(
        model, dummy_input, str(onnx_path),
        export_params=True, opset_version=17, do_constant_folding=True,
        input_names=["features"], output_names=["reconstruction"],
        dynamic_axes={"features": {0: "batch_size"}, "reconstruction": {0: "batch_size"}}
    )
    print(f"[EXPORTED] → {onnx_path}")
    print(f"   Input shape: {dummy_input.shape}")


def main():
    with open("configs/model.yaml") as f:
        cfg = yaml.safe_load(f)
    with open("configs/data.yaml") as f:
        data_cfg = yaml.safe_load(f)

    Path(data_cfg["paths"]["models"]).mkdir(exist_ok=True)
    export_anomaly(cfg, data_cfg)

    print("\n✅ Export complete! ONNX files are ready for deployment.")
    print("   Copy .onnx + scaler.joblib to your inference node (Pi/Synology)")


if __name__ == "__main__":
    main()
