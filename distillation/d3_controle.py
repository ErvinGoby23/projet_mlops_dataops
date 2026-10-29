"""
D3 — CONTRÔLE : la v1 continue son entraînement SANS professeur

Question : le gain de d2 vient-il du professeur, ou juste de l'entraînement supplémentaire ?
Même chose que d2 (départ = v1, même recette), mais SANS distill_model.
À lancer dans le venv GPU. Si model/best_continue.pt existe déjà, l'entraînement est sauté.
"""
import shutil
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402
from ultralytics import YOLO  # noqa: E402

from optimisation.common import (CFG, P, ROOT, banner, eval_yaml, evaluate,  # noqa: E402
                                 save_json, size_mb)

OUT = ROOT / "model" / "best_continue.pt"
DISTILL = ROOT / "model" / "best_distill.pt"
NAME = "continue_v8n"
IMGSZ = 640


def train():
    t = CFG["training"]
    device = 0 if torch.cuda.is_available() else "cpu"
    print(f"Contrôle (sans professeur) sur : {torch.cuda.get_device_name(0) if device == 0 else 'CPU'}")
    model = YOLO(str(P["model_pt"]))                  # même départ que d2 : la v1
    model.train(data=str(P["data"]), epochs=t["epochs"], imgsz=t["imgsz"], batch=t["batch"],
                patience=t["patience"], fliplr=0.0, device=device, workers=2,
                project=str(P["runs"]), name=NAME, exist_ok=True)   # PAS de distill_model
    best = Path(model.trainer.save_dir) / "weights" / "best.pt"
    shutil.copy(best, OUT)
    print(f"✅ Contrôle copié dans {OUT}")


def main():
    if OUT.exists():
        print(f"ℹ️  {OUT.name} existe déjà : entraînement sauté.")
    else:
        train()

    data_eval = eval_yaml()
    models = {"v1": P["model_pt"], "v1_continuee_sans_prof": OUT, "v1_distillee": DISTILL}
    res = {"imgsz": IMGSZ}
    for name, path in models.items():
        t = evaluate(path, IMGSZ, tag=f"{name}_test", project="distill_d3")
        e = evaluate(path, IMGSZ, data=data_eval, split="test", tag=f"{name}_216", project="distill_d3")
        res[name] = {"taille_mo": size_mb(path), "mAP50_test": t[0], "mAP50-95_test": t[1],
                     "mAP50_216img": e[0], "mAP50-95_216img": e[1]}
    save_json("distill_d3.json", res)

    lines = banner(f"D3 — Contrôle : le gain vient-il du professeur ? ({IMGSZ})") + [
        f"{'':24}{'mAP50':>8}{'50-95':>8}{'mAP50':>9}{'50-95':>8}",
        f"{'':24}{'test (72)':>16}{'216 images':>17}"]
    for label, key in [("v1", "v1"), ("v1 continuée (sans prof)", "v1_continuee_sans_prof"),
                       ("v1 distillée (avec prof)", "v1_distillee")]:
        r = res[key]
        lines.append(f"{label:24}{r['mAP50_test']:>8.3f}{r['mAP50-95_test']:>8.3f}"
                     f"{r['mAP50_216img']:>9.3f}{r['mAP50-95_216img']:>8.3f}")
    c, d = res["v1_continuee_sans_prof"], res["v1_distillee"]
    lines.append(f"Effet du professeur seul (distillée − continuée), 216 images : "
                 f"mAP50 {d['mAP50_216img'] - c['mAP50_216img']:+.3f} | "
                 f"mAP50-95 {d['mAP50-95_216img'] - c['mAP50-95_216img']:+.3f}")
    print("\n".join(lines))


if __name__ == "__main__":      # obligatoire sous Windows (multiprocessing)
    main()