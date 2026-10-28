"""
Entraînement du modèle (données -> modèle), lu depuis config.yaml.

    $env:ROBOFLOW_API_KEY="ta_cle"     (PowerShell, seulement si le dataset n'est pas déjà là)
    python training/train.py                                   # v1 : dataset seul -> model/best.pt
    python training/train.py --data datasets_ervin/data_v2.yaml --out model/best_v2.pt --name train_v2

Télécharge le dataset si besoin, fine-tune YOLOv8n sur GPU (ou CPU), évalue sur le jeu de test
et copie le meilleur modèle dans --out (par défaut model/best.pt).
"""
import argparse
import os
import shutil
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402
from ultralytics import YOLO  # noqa: E402

from optimisation.common import CFG, P, ROOT  # noqa: E402


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
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(P["data"]), help="fichier data.yaml à utiliser")
    ap.add_argument("--out", default=str(P["model_pt"]), help="où copier le meilleur modèle")
    ap.add_argument("--name", default="train", help="nom du dossier dans runs/")
    args = ap.parse_args()

    data = Path(args.data) if Path(args.data).is_absolute() else ROOT / args.data
    out = Path(args.out) if Path(args.out).is_absolute() else ROOT / args.out

    if out.exists() and out.resolve() == P["model_pt"].resolve() and args.data != str(P["data"]):
        sys.exit("❌ Refusé : ces données écraseraient model/best.pt (la v1). Utilise --out.")

    ensure_dataset()   # le dataset David Lee sert aussi à la v2
    t = CFG["training"]
    device = 0 if torch.cuda.is_available() else "cpu"
    print(f"Entraînement sur : {torch.cuda.get_device_name(0) if device == 0 else 'CPU'}")
    print(f"Données : {data}\nSortie  : {out}")

    model = YOLO(t["base_model"])   # on repart de yolov8n.pt (COCO), comme la v1
    model.train(data=str(data), epochs=t["epochs"], imgsz=t["imgsz"], batch=t["batch"],
                patience=t["patience"], fliplr=0.0,   # pas de miroir : change le sens d'un signe
                device=device, workers=2, project=str(P["runs"]), name=args.name, exist_ok=True)

    best = Path(model.trainer.save_dir) / "weights" / "best.pt"
    m = YOLO(str(best)).val(data=str(data), split="test", device=device, workers=2)
    print(f"mAP50 test : {m.box.map50:.3f} | mAP50-95 : {m.box.map:.3f}")

    out.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(best, out)
    print(f"✅ Modèle copié dans {out}")


if __name__ == "__main__":      # obligatoire sous Windows (multiprocessing)
    main()