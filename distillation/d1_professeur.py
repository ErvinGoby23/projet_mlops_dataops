"""
D1 — Entraîner le PROFESSEUR (YOLOv8s, ~11 M paramètres) sur le dataset ASL

Question : un modèle plus gros fait-il mieux que la v1 (YOLOv8n, ~3 M) ?
Il doit dépasser la v1, sinon il n'a rien à apprendre à l'élève.
Même recette que la v1 (config.yaml, pas de miroir). À lancer dans le venv GPU.
Si model/teacher_v8s.pt existe déjà, l'entraînement est sauté (comme export_onnx).

Évaluation sur 2 jeux :
  - test (72 images) : même jeu que la v1 (0,924 / 0,765)
  - valid + test (216 images) : plus robuste, comme E3 / E4 / E6
    (⚠️ valid a servi à choisir la meilleure epoch -> légèrement optimiste ; sert à COMPARER)
"""
import shutil
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402
from ultralytics import YOLO  # noqa: E402

from optimisation.common import (CFG, P, ROOT, banner, eval_yaml, evaluate,  # noqa: E402
                                 save_json, size_mb)

BASE = "yolov8s.pt"
OUT = ROOT / "model" / "teacher_v8s.pt"
NAME = "teacher_v8s"
IMGSZ = 640


def train():
    if not P["data"].exists():
        sys.exit("❌ Dataset absent : lance d'abord training/train.py (il le télécharge).")
    t = CFG["training"]
    device = 0 if torch.cuda.is_available() else "cpu"
    print(f"Entraînement du professeur sur : {torch.cuda.get_device_name(0) if device == 0 else 'CPU'}")
    model = YOLO(BASE)
    model.train(data=str(P["data"]), epochs=t["epochs"], imgsz=t["imgsz"], batch=t["batch"],
                patience=t["patience"], fliplr=0.0, device=device, workers=2,
                project=str(P["runs"]), name=NAME, exist_ok=True)
    best = Path(model.trainer.save_dir) / "weights" / "best.pt"
    OUT.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(best, OUT)
    print(f"✅ Professeur copié dans {OUT}")


def main():
    if OUT.exists():
        print(f"ℹ️  {OUT.name} existe déjà : entraînement sauté.")
    else:
        train()

    data_eval = eval_yaml()
    models = {"v1_yolov8n": P["model_pt"], "professeur_yolov8s": OUT}
    res = {"imgsz": IMGSZ}
    for name, path in models.items():
        t = evaluate(path, IMGSZ, tag=f"{name}_test", project="distill_d1")
        e = evaluate(path, IMGSZ, data=data_eval, split="test", tag=f"{name}_216", project="distill_d1")
        res[name] = {"taille_mo": size_mb(path),
                     "mAP50_test": t[0], "mAP50-95_test": t[1],
                     "mAP50_216img": e[0], "mAP50-95_216img": e[1], "ap50_par_lettre": e[2]}

    v1, prof = res["v1_yolov8n"], res["professeur_yolov8s"]
    res["professeur_meilleur"] = prof["mAP50-95_216img"] > v1["mAP50-95_216img"]
    save_json("distill_d1.json", res)

    lines = banner(f"D1 — Professeur YOLOv8s vs v1 YOLOv8n ({IMGSZ})") + [
        f"{'':14}{'Mo':>7}{'mAP50':>8}{'50-95':>8}{'mAP50':>9}{'50-95':>8}",
        f"{'':14}{'':>7}{'test (72)':>16}{'216 images':>17}"]
    for label, r in [("v1 (v8n)", v1), ("Prof (v8s)", prof)]:
        lines.append(f"{label:14}{r['taille_mo']:>7.2f}{r['mAP50_test']:>8.3f}{r['mAP50-95_test']:>8.3f}"
                     f"{r['mAP50_216img']:>9.3f}{r['mAP50-95_216img']:>8.3f}")
    lines.append(f"{'Écart':14}{'':>7}{prof['mAP50_test'] - v1['mAP50_test']:>+8.3f}"
                 f"{prof['mAP50-95_test'] - v1['mAP50-95_test']:>+8.3f}"
                 f"{prof['mAP50_216img'] - v1['mAP50_216img']:>+9.3f}"
                 f"{prof['mAP50-95_216img'] - v1['mAP50-95_216img']:>+8.3f}")
    lines.append("✅ Le professeur est meilleur (mAP50-95, 216 images) : on peut distiller."
                 if res["professeur_meilleur"]
                 else "⚠️  Le professeur n'est pas meilleur que la v1 (mAP50-95, 216 images).")
    print("\n".join(lines))


if __name__ == "__main__":      # obligatoire sous Windows (multiprocessing)
    main()