

import os
import shutil
import sys
from pathlib import Path

# Ajout du dossier parent au chemin systeme pour l'importation des modules internes
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from ultralytics import YOLO

from optimisation.common import CFG, P


def ensure_dataset():
    """Verifie la presence du dataset local ou le telecharge automatiquement via Roboflow."""
    if P["data"].exists():
        return
        
    key = os.environ.get("ROBOFLOW_API_KEY")
    if not key:
        sys.exit('Erreur: Dataset absent. Definissez $env:ROBOFLOW_API_KEY="ta_cle" puis relancez.')
        
    from roboflow import Roboflow
    d = CFG["dataset"]
    
    # Telechargement du dataset depuis le workspace Roboflow indique dans config.yaml
    Roboflow(api_key=key).workspace(d["roboflow_workspace"]).project(d["roboflow_project"]) \
        .version(d["roboflow_version"]).download("yolov8", location=str(P["data"].parent))


def main():
    # 1. Verification et preparation du jeu de donnees
    ensure_dataset()
    t = CFG["training"]
    
    # Configuration du peripherique d'execution (GPU Nvidia si disponible, sinon CPU)
    device = 0 if torch.cuda.is_available() else "cpu"
    print(f"Entrainement sur : {torch.cuda.get_device_name(0) if device == 0 else 'CPU'}")

    # 2. Initialisation et lancement du fine-tuning YOLOv8
    model = YOLO(t["base_model"])
    model.train(
        data=str(P["data"]),
        epochs=t["epochs"],
        imgsz=t["imgsz"],
        batch=t["batch"],
        patience=t["patience"],
        fliplr=0.0,   # Inactivation de l'augmentation par miroir horizontal (modifierait le sens des signes)
        device=device,
        workers=2,
        project=str(P["runs"]),
        name="train",
        exist_ok=True
    )

    # 3. Evaluation des performances du meilleur checkpoint (best.pt) sur le split de test
    best = Path(model.trainer.save_dir) / "weights" / "best.pt"
    m = YOLO(str(best)).val(data=str(P["data"]), split="test", device=device, workers=2)
    print(f"mAP50 test : {m.box.map50:.3f} | mAP50-95 : {m.box.map:.3f}")

    # 4. Copie du modele entraine vers le repertoire cible de production/optimisation
    P["model_pt"].parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(best, P["model_pt"])
    print(f"OK: Modele copie dans {P['model_pt']}")


if __name__ == "__main__":
    # Condition requise sous Windows pour assurer le bon fonctionnement du multiprocessing
    main()