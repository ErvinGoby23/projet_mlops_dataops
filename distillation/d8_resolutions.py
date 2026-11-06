import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ultralytics import YOLO  # noqa: E402

from optimisation.common import (CFG, P, ROOT, banner, bench_onnx, calibration_files,  # noqa: E402
                                 eval_yaml, evaluate, quantize_static, save_json, size_mb)

DISTILL = ROOT / "model" / "best_distill.pt"
RESOLUTIONS = [640, 416, 320]


def export_distill(size):
    dst = P["exports"] / f"distill_fp32_{size}.onnx"
    if not dst.exists():
        p = YOLO(str(DISTILL)).export(format="onnx", imgsz=size, dynamic=True, simplify=True, device="cpu")
        Path(p).replace(dst)
    return dst


def main():
    if not DISTILL.exists():
        sys.exit(f"{DISTILL} absent")
    calib = calibration_files(CFG["quantization"]["calib_images"])
    data_eval = eval_yaml()
    res = {}
    for size in RESOLUTIONS:
        fp32 = export_distill(size)
        int8 = P["exports"] / f"distill_int8_{size}_percentile.onnx"
        if not int8.exists():
            quantize_static(fp32, int8, size, calib, method="percentile",
                            extra_options={"CalibPercentile": 99.999})
        for prec, path in (("FP32", fp32), ("INT8", int8)):
            name = f"{size}_{prec}"
            t = evaluate(path, size, tag=f"{name}_test", project="distill_d8")
            e = evaluate(path, size, data=data_eval, split="test", tag=f"{name}_216", project="distill_d8")
            res[name] = {"fichier": path.name, "taille_mo": size_mb(path),
                         "mAP50_test": t[0], "mAP50_216img": e[0], "mAP50-95_216img": e[1],
                         "ms_b1": bench_onnx(path, size, batches=[1])[1]}
    save_json("distill_d8.json", res)

    lines = banner("D8 — Distillée : résolution × quantization (ONNX CPU)") + [
        f"{'':12}{'Mo':>7}{'mAP50':>8}{'216img':>8}{'50-95':>8}{'ms b1':>8}"]
    for name, r in res.items():
        lines.append(f"{name:12}{r['taille_mo']:>7.2f}{r['mAP50_test']:>8.3f}{r['mAP50_216img']:>8.3f}"
                     f"{r['mAP50-95_216img']:>8.3f}{r['ms_b1']:>8.1f}")
    print("\n".join(lines))


if __name__ == "__main__":
    main()