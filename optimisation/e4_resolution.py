"""
Expérience 4 — 2e optimisation : résolution d'entrée (640 / 416 / 320) × FP32 / INT8

Question : réduire le nombre de pixels accélère-t-il, et à quel coût en précision ?
(Les images du dataset sont nativement en 416 : agrandir en 640 n'ajoute aucune information.)
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from optimisation.common import (CFG, P, banner, bench_onnx, calibration_files, eval_yaml,  # noqa: E402
                                 evaluate, export_onnx, quantize_static, save_json, size_mb)


def main():
    calib = calibration_files(CFG["quantization"]["calib_images"])
    data_eval = eval_yaml()
    models = {}
    for size in CFG["resolutions"]:
        fp32 = export_onnx(size)
        int8 = quantize_static(fp32, P["exports"] / f"asl_int8_static_{size}.onnx", size, calib)
        models[f"{size}_FP32"], models[f"{size}_INT8"] = (fp32, size), (int8, size)

    res = {}
    for name, (path, size) in models.items():
        t = evaluate(path, size, tag=f"{name}_test", project="e4")
        e = evaluate(path, size, data=data_eval, split="test", tag=f"{name}_216", project="e4")
        res[name] = {"resolution": size, "taille_mo": size_mb(path),
                     "mAP50_test": t[0], "mAP50-95_test": t[1],
                     "mAP50_216img": e[0], "mAP50-95_216img": e[1], "ap50_par_lettre": e[2],
                     "ms_par_image": bench_onnx(path, size, batches=[1, 8])}

    base_name = f"{CFG['resolutions'][0]}_FP32"
    ref, agg = res[base_name]["ap50_par_lettre"], res[f"{CFG['resolutions'][-1]}_INT8"]["ap50_par_lettre"]
    res["_lettres_plus_degradees_320_INT8"] = sorted(
        ((l, agg.get(l, 0.0) - ap) for l, ap in ref.items()), key=lambda x: x[1])[:5]
    save_json("etape4.json", res)

    base = res[base_name]
    lines = banner("E4 — Résolution × quantization (CPU)") + [
        f"{'':12}{'Mo':>7}{'mAP50':>8}{'216img':>8}{'ms b1':>8}{'Accél.':>8}{'Perte':>8}"]
    for name in models:
        r = res[name]
        accel = "x{:.2f}".format(base["ms_par_image"][1] / r["ms_par_image"][1])
        lines.append(f"{name:12}{r['taille_mo']:>7.2f}{r['mAP50_test']:>8.3f}{r['mAP50_216img']:>8.3f}"
                     f"{r['ms_par_image'][1]:>8.1f}{accel:>8}{r['mAP50_216img'] - base['mAP50_216img']:>+8.3f}")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
