"""
D4 — Quantifier la v1 DISTILLÉE avec la même recette que la v1 (E4), à 416

Question : le gain de la distillation survit-il à la quantization INT8 ?
Recette identique à la v1 : QDQ statique, MinMax, per-channel, tête FP32, 100 images de calibration (train).
La v1 n'est PAS modifiée : ses fichiers (model/best.pt, exports/asl_*.onnx) sont seulement lus.
Nouveaux fichiers : exports/distill_fp32_416.onnx, exports/distill_int8_416.onnx
À lancer dans le venv CPU (celui des expériences E1 à E6).
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ultralytics import YOLO  # noqa: E402

from optimisation.common import (CFG, P, ROOT, banner, bench_onnx, calibration_files,  # noqa: E402
                                 eval_yaml, evaluate, quantize_static, save_json, size_mb)

DISTILL = ROOT / "model" / "best_distill.pt"
IMGSZ = 416


def export_distill(size):
    """Exporte le modèle distillé en ONNX FP32 (réutilise le fichier s'il existe)."""
    dst = P["exports"] / f"distill_fp32_{size}.onnx"
    if not dst.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        p = YOLO(str(DISTILL)).export(format="onnx", imgsz=size, dynamic=True, simplify=True, device="cpu")
        Path(p).replace(dst)
    return dst


def main():
    if not DISTILL.exists():
        sys.exit("❌ model/best_distill.pt absent : lance d'abord distillation/d2_eleve.py")

    fp32 = export_distill(IMGSZ)
    int8 = P["exports"] / f"distill_int8_{IMGSZ}.onnx"
    if not int8.exists():
        calib = calibration_files(CFG["quantization"]["calib_images"])
        quantize_static(fp32, int8, IMGSZ, calib)        # même recette que la v1 (tête FP32)

    models = {
        "v1_FP32": P["exports"] / f"asl_fp32_{IMGSZ}.onnx",
        "v1_INT8 (prod)": P["exports"] / f"asl_int8_static_{IMGSZ}.onnx",
        "distill_FP32": fp32,
        "distill_INT8": int8,
    }
    data_eval = eval_yaml()
    res = {"imgsz": IMGSZ}
    for name, path in models.items():
        if not path.exists():
            print(f"⚠️  {path.name} absent : ignoré")
            continue
        print(f"→ {name}")
        tag = name.split()[0]
        t = evaluate(path, IMGSZ, tag=f"{tag}_test", project="distill_d4")
        e = evaluate(path, IMGSZ, data=data_eval, split="test", tag=f"{tag}_216", project="distill_d4")
        res[name] = {"fichier": path.name, "taille_mo": size_mb(path),
                     "mAP50_test": t[0], "mAP50-95_test": t[1],
                     "mAP50_216img": e[0], "mAP50-95_216img": e[1],
                     "ms_b1": bench_onnx(path, IMGSZ, batches=[1])[1]}
    save_json("distill_d4.json", res)

    lines = banner(f"D4 — Distillation + quantization (ONNX CPU, {IMGSZ})") + [
        f"{'':16}{'Mo':>7}{'mAP50':>8}{'50-95':>8}{'mAP50':>9}{'50-95':>8}{'ms b1':>8}",
        f"{'':16}{'':>7}{'test (72)':>16}{'216 images':>17}"]
    for name in models:
        if name in res:
            r = res[name]
            lines.append(f"{name:16}{r['taille_mo']:>7.2f}{r['mAP50_test']:>8.3f}{r['mAP50-95_test']:>8.3f}"
                         f"{r['mAP50_216img']:>9.3f}{r['mAP50-95_216img']:>8.3f}{r['ms_b1']:>8.1f}")
    if "v1_INT8 (prod)" in res and "distill_INT8" in res:
        a, b = res["v1_INT8 (prod)"], res["distill_INT8"]
        lines.append(f"Distillée INT8 − v1 INT8 (216 images) : mAP50 {b['mAP50_216img'] - a['mAP50_216img']:+.3f}"
                     f" | mAP50-95 {b['mAP50-95_216img'] - a['mAP50-95_216img']:+.3f}")
    print("\n".join(lines))


if __name__ == "__main__":
    main()