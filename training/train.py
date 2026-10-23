"""
Entraînement du modèle (données -> modèle), lu depuis config.yaml.

    $env:ROBOFLOW_API_KEY="ta_cle"     (PowerShell, seulement si le dataset n'est pas déjà là)
    python training/train.py

Télécharge le dataset si besoin, fine-tune YOLOv8n sur GPU (ou CPU), évalue sur le jeu de test
et copie le meilleur modèle dans model/best.pt (point de départ de la quantization).
"""
import os
import shutil
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402
from ultralytics import YOLO  # noqa: E402

from optimisation.common import CFG, P  # noqa: E402


def ensure_dataset():
    if P["data"].exists():
        return
    key = os.environ.get("ROBOFLOW_API_KEY")
    if not key:
        sys.exit('❌ Dataset absent : définis $env:ROBOFLOW_API_KEY="ta_cle" puis relance.')
    from roboflow import Roboflow
    d = CFG["dataset"]
    Roboflow(api_key=key).workspace(d["roboflow_workspace"]).project(d["roboflow_project"]) \
        .version(d["roboflow_version"]).download("yolov8", location=str(P["data"].parent))


def main():
    ensure_dataset()
    t = CFG["training"]
    device = 0 if torch.cuda.is_available() else "cpu"
    print(f"Entraînement sur : {torch.cuda.get_device_name(0) if device == 0 else 'CPU'}")

    model = YOLO(t["base_model"])
    model.train(data=str(P["data"]), epochs=t["epochs"], imgsz=t["imgsz"], batch=t["batch"],
                patience=t["patience"], fliplr=0.0,   # pas de miroir : change le sens d'un signe
                device=device, workers=2, project=str(P["runs"]), name="train", exist_ok=True)

    best = Path(model.trainer.save_dir) / "weights" / "best.pt"
    m = YOLO(str(best)).val(data=str(P["data"]), split="test", device=device, workers=2)
    print(f"mAP50 test : {m.box.map50:.3f} | mAP50-95 : {m.box.map:.3f}")

    P["model_pt"].parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(best, P["model_pt"])
    print(f"✅ Modèle copié dans {P['model_pt']}")


if __name__ == "__main__":      # obligatoire sous Windows (multiprocessing)
    main()
