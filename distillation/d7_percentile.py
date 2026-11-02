"""
D7 — Requantifier la v1 DISTILLÉE avec la calibration Percentile (recette retenue après E8).

Même recette que D4 (QDQ statique, per-channel, tête FP32, 100 images de calibration du train),
seule la calibration change : Percentile 99,999 au lieu de MinMax.
Lit exports/distill_fp32_416.onnx (créé par D4) ; écrit exports/distill_int8_416_percentile.onnx.
D4 et ses fichiers ne sont pas modifiés.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from optimisation.common import (CFG, P, banner, bench_onnx, calibration_files,  # noqa: E402
                                 eval_yaml, evaluate, quantize_static, save_json, size_mb)

IMGSZ = 416


def main():
    fp32 = P["exports"] / f"distill_fp32_{IMGSZ}.onnx"
    if not fp32.exists():
        sys.exit(f"❌ {fp32} absent : lance d'abord d4_quantif_distill.py")
    calib = calibration_files(CFG["quantization"]["calib_images"])
    int8 = quantize_static(fp32, P["exports"] / f"distill_int8_{IMGSZ}_percentile.onnx", IMGSZ, calib,
                           method="percentile", extra_options={"CalibPercentile": 99.999})

    data_eval = eval_yaml()
    res = {}
    for name, path in {"distill_FP32": fp32, "distill_INT8_percentile": int8}.items():
        t = evaluate(path, IMGSZ, tag=f"{name}_test", project="distill_d7")
        e = evaluate(path, IMGSZ, data=data_eval, split="test", tag=f"{name}_216", project="distill_d7")
        res[name] = {"fichier": path.name, "taille_mo": size_mb(path),
                     "mAP50_test": t[0], "mAP50-95_test": t[1],
                     "mAP50_216img": e[0], "mAP50-95_216img": e[1],
                     "ms_b1": bench_onnx(path, IMGSZ, batches=[1])[1]}
    save_json("distill_d7.json", res)

    lines = banner(f"D7 — Distillée, calibration Percentile (ONNX CPU, {IMGSZ})") + [
        f"{'':26}{'Mo':>7}{'mAP50':>8}{'216img':>8}{'50-95':>8}{'ms b1':>8}"]
    for name, r in res.items():
        lines.append(f"{name:26}{r['taille_mo']:>7.2f}{r['mAP50_test']:>8.3f}{r['mAP50_216img']:>8.3f}"
                     f"{r['mAP50-95_216img']:>8.3f}{r['ms_b1']:>8.1f}")
    print("\n".join(lines))


if __name__ == "__main__":
    main()