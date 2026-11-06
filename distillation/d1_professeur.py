import shutil
import sys
from pathlib import Path

# Ajout du dossier parent au chemin systeme pour l'importation des modules internes
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from ultralytics import YOLO

from optimisation.common import (
    CFG,
    P,
    ROOT,
    banner,
    eval_yaml,
    evaluate,
    save_json,
    size_mb,
)

# Constants et chemins de destination du professeur
BASE = "yolov8s.pt"
OUT = ROOT / "model" / "teacher_v8s.pt"
NAME = "teacher_v8s"
IMGSZ = 640


def train():
    """Fine-tune le modele professeur YOLOv8s sur le jeu de donnees d'apprentissage."""
    if not P["data"].exists():
        sys.exit("Erreur: Dataset absent. Lancez d'abord training/train.py pour le telecharger.")
        
    t = CFG["training"]
    device = 0 if torch.cuda.is_available() else "cpu"
    print(f"Entrainement du professeur sur : {torch.cuda.get_device_name(0) if device == 0 else 'CPU'}")
    
    # Chargement et entrainement de YOLOv8s sans augmentation par miroir horizontal
    model = YOLO(BASE)
    model.train(
        data=str(P["data"]),
        epochs=t["epochs"],
        imgsz=t["imgsz"],
        batch=t["batch"],
        patience=t["patience"],
        fliplr=0.0,
        device=device,
        workers=2,
        project=str(P["runs"]),
        name=NAME,
        exist_ok=True,
    )
    
    # Extraction et copie du meilleur checkpoint
    best = Path(model.trainer.save_dir) / "weights" / "best.pt"
    OUT.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(best, OUT)
    print(f"OK: Professeur copie dans {OUT}")


def main():
    # 1. Verification de l'existence du modele professeur
    if OUT.exists():
        print(f"Info: {OUT.name} existe deja. Entrainement saute.")
    else:
        train()

    # 2. Evaluation comparative sur le jeu de test et sur le jeu elargi (216 images)
    data_eval = eval_yaml()
    models = {"v1_yolov8n": P["model_pt"], "professeur_yolov8s": OUT}
    res = {"imgsz": IMGSZ}
    
    for name, path in models.items():
        t = evaluate(path, IMGSZ, tag=f"{name}_test", project="distill_d1")
        e = evaluate(path, IMGSZ, data=data_eval, split="test", tag=f"{name}_216", project="distill_d1")
        res[name] = {
            "taille_mo": size_mb(path),
            "mAP50_test": t[0],
            "mAP50-95_test": t[1],
            "mAP50_216img": e[0],
            "mAP50-95_216img": e[1],
            "ap50_par_lettre": e[2],
        }

    # 3. Validation de la superiorite du professeur par rapport a l'eleve (v1)
    v1, prof = res["v1_yolov8n"], res["professeur_yolov8s"]
    res["professeur_meilleur"] = prof["mAP50-95_216img"] > v1["mAP50-95_216img"]
    save_json("distill_d1.json", res)

    # 4. Affichage du compte-rendu synthetique
    lines = banner(f"D1 — Professeur YOLOv8s vs v1 YOLOv8n ({IMGSZ})") + [
        f"{'':14}{'Mo':>7}{'mAP50':>8}{'50-95':>8}{'mAP50':>9}{'50-95':>8}",
        f"{'':14}{'':>7}{'test (72)':>16}{'216 images':>17}",
    ]
    for label, r in [("v1 (v8n)", v1), ("Prof (v8s)", prof)]:
        lines.append(
            f"{label:14}{r['taille_mo']:>7.2f}{r['mAP50_test']:>8.3f}{r['mAP50-95_test']:>8.3f}"
            f"{r['mAP50_216img']:>9.3f}{r['mAP50-95_216img']:>8.3f}"
        )
    lines.append(
        f"{'Ecart':14}{'':>7}{prof['mAP50_test'] - v1['mAP50_test']:>+8.3f}"
        f"{prof['mAP50-95_test'] - v1['mAP50-95_test']:>+8.3f}"
        f"{prof['mAP50_216img'] - v1['mAP50_216img']:>+9.3f}"
        f"{prof['mAP50-95_216img'] - v1['mAP50-95_216img']:>+8.3f}"
    )
    lines.append(
        "OK: Le professeur est meilleur (mAP50-95, 216 images) : distillation validee."
        if res["professeur_meilleur"]
        else "Attention: Le professeur n'est pas meilleur que la v1 (mAP50-95, 216 images)."
    )
    print("\n".join(lines))


if __name__ == "__main__":
    # Condition requise sous Windows pour la gestion du multiprocessing PyTorch
    main()