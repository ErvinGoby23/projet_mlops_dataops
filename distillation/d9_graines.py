"""
D9 — Robustesse à la graine : distillation (d2) vs contrôle sans professeur (d3), graines 0, 1, 2.

Même recette que d2 / d3 (départ = v1, config.yaml, pas de miroir) ; seule la graine change.
Graine 0 = modèles déjà entraînés (best_distill.pt, best_continue.pt).

Étape 1 (venv GPU) : python distillation/d9_graines.py --etape train
Étape 2 (venv CPU) : python distillation/d9_graines.py --etape eval --images datasets_externe/confirmation_50
  -> pour chaque modèle : export ONNX 416, INT8 Percentile, exactitude sur les images externes (FP32 et INT8)
"""
import argparse
import shutil
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
from ultralytics import YOLO  # noqa: E402

from optimisation.common import CFG, P, ROOT, banner, calibration_files, quantize_static, save_json  # noqa: E402

TEACHER = ROOT / "model" / "teacher_v8s.pt"
GRAINES = [0, 1, 2]
IMGSZ = 416


def chemin(methode, graine):
    if graine == 0:
        return ROOT / "model" / ("best_distill.pt" if methode == "distill" else "best_continue.pt")
    return ROOT / "model" / f"best_{methode}_s{graine}.pt"


def train():
    import torch
    t = CFG["training"]
    device = 0 if torch.cuda.is_available() else "cpu"
    for g in GRAINES[1:]:
        for methode in ("distill", "continue"):
            out = chemin(methode, g)
            if out.exists():
                print(f"ℹ️  {out.name} existe déjà : sauté")
                continue
            extra = {"distill_model": str(TEACHER)} if methode == "distill" else {}
            model = YOLO(str(P["model_pt"]))                       # départ = v1, comme d2 / d3
            model.train(data=str(P["data"]), epochs=t["epochs"], imgsz=t["imgsz"], batch=t["batch"],
                        patience=t["patience"], fliplr=0.0, device=device, workers=2,
                        seed=g, deterministic=True,
                        project=str(P["runs"]), name=f"{methode}_s{g}", exist_ok=True, **extra)
            shutil.copy(Path(model.trainer.save_dir) / "weights" / "best.pt", out)
            print(f"✅ {out.name}")


def exactitude(model_path, files, gts):
    m = YOLO(str(model_path), task="detect")
    ok = []
    for f, g in zip(files, gts):
        r = m.predict(str(f), imgsz=IMGSZ, conf=0.25, device="cpu", verbose=False)[0]
        pred = int(r.boxes.cls[int(r.boxes.conf.argmax())]) if len(r.boxes) else None
        ok.append(int(pred == g))
    return float(np.mean(ok))


def evaluer(images):
    calib = calibration_files(CFG["quantization"]["calib_images"])
    names = YOLO(str(P["model_pt"])).names
    idx = {v.upper(): k for k, v in names.items()}
    files, gts = [], []
    for d in sorted(Path(images).iterdir()):
        if d.is_dir() and d.name.upper() in idx:
            for f in sorted(d.glob("*.jpg")):
                files.append(f); gts.append(idx[d.name.upper()])
    print(f"{len(files)} images")

    res = {"images": images, "n": len(files), "modeles": {}}
    for methode in ("distill", "continue"):
        for g in GRAINES:
            pt = chemin(methode, g)
            if not pt.exists():
                print(f"⚠️  {pt.name} absent : ignoré"); continue
            fp32 = P["exports"] / f"d9_{methode}_s{g}_fp32_{IMGSZ}.onnx"
            if not fp32.exists():
                Path(YOLO(str(pt)).export(format="onnx", imgsz=IMGSZ, dynamic=True, simplify=True,
                                          device="cpu")).replace(fp32)
            int8 = P["exports"] / f"d9_{methode}_s{g}_int8_{IMGSZ}_percentile.onnx"
            if not int8.exists():
                quantize_static(fp32, int8, IMGSZ, calib, method="percentile",
                                extra_options={"CalibPercentile": 99.999})
            r = {"fp32": exactitude(fp32, files, gts), "int8": exactitude(int8, files, gts)}
            res["modeles"][f"{methode}_s{g}"] = r
            print(f"  {methode} graine {g} : FP32 {r['fp32']:.3f} | INT8 {r['int8']:.3f}")

    lines = banner(f"D9 — Graines 0/1/2 : distillation vs contrôle ({IMGSZ} px, {len(files)} images externes)")
    for methode, label in (("distill", "Distillée (avec prof)"), ("continue", "Contrôle (sans prof)")):
        for prec in ("fp32", "int8"):
            v = [res["modeles"][k][prec] for k in res["modeles"] if k.startswith(methode)]
            if v:
                res[f"{methode}_{prec}_moyenne"], res[f"{methode}_{prec}_ecart_type"] = float(np.mean(v)), float(np.std(v))
                lines.append(f"{label:22} {prec.upper()} : " + " | ".join(f"{x:.3f}" for x in v)
                             + f"   -> moyenne {np.mean(v):.3f} ± {np.std(v):.3f}")
    save_json("d9_graines.json", res)
    print("\n".join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--etape", choices=["train", "eval"], required=True)
    ap.add_argument("--images", default="datasets_externe/confirmation_50")
    a = ap.parse_args()
    train() if a.etape == "train" else evaluer(a.images)


if __name__ == "__main__":      # obligatoire sous Windows (multiprocessing)
    main()