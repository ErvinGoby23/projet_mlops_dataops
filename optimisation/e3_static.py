"""
Expérience 3 — Quantization post-training STATIQUE (calibrée), 3 variantes à 640

  A. MinMax,     tout quantifié          -> version naïve
  B. MinMax,     tête de détection FP32  -> précision mixte (recette retenue)
  C. Percentile, tête de détection FP32  -> les outliers comptent-ils encore ?

Question : calculer les échelles des activations à l'avance permet-il d'accélérer sans perdre ?

Note v8m : la variante C (Percentile) garde en mémoire toutes les activations des images
de calibration -> manque de RAM avec 100 images sur le v8m. Elle est calibrée sur 25 images.
Les variantes déjà quantifiées (fichier présent dans exports) sont réutilisées.
"""
import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from optimisation.common import (CFG, P, banner, batch_of, bench_onnx, calibration_files,  # noqa: E402
                                 eval_yaml, evaluate, export_onnx, images_of, quantize_static,
                                 save_json, score_gap, size_mb, test_split)

IMGSZ = 640
CALIB_PERCENTILE = 25          # images de calibration pour Percentile (limite de RAM avec le v8m)
VARIANTS = {
    "A_minmax_tout": dict(method="minmax", protect_head=False, extra={}),
    "B_minmax_tete_fp32": dict(method="minmax", protect_head=True, extra={}),
    "C_percentile_tete_fp32": dict(method="percentile", protect_head=True,
                                   extra={"CalibPercentile": 99.999}),
}


def main():
    fp32 = export_onnx(IMGSZ)
    calib = calibration_files(CFG["quantization"]["calib_images"])
    paths = {}
    for name, v in VARIANTS.items():
        t0 = time.perf_counter()
        dst = P["exports"] / f"asl_int8_static_{name}.onnx"
        if dst.exists():
            paths[name] = dst
            print(f"↩️  {name} déjà quantifié, réutilisé ({size_mb(dst):.2f} Mo)")
            continue
        files = calib[:CALIB_PERCENTILE] if v["method"] == "percentile" else calib
        paths[name] = quantize_static(fp32, dst, IMGSZ, files, method=v["method"],
                                      protect_head=v["protect_head"], extra_options=v["extra"])
        print(f"✅ {name} ({size_mb(paths[name]):.2f} Mo, {len(files)} images de calibration, "
              f"{time.perf_counter() - t0:.0f} s)")

    data_eval = eval_yaml()
    imgs = batch_of(images_of(test_split())[:16], IMGSZ)
    res = {}
    for name, p in {"FP32": fp32, **paths}.items():
        t = evaluate(p, IMGSZ, tag=f"{name}_test", project="e3")
        e = evaluate(p, IMGSZ, data=data_eval, split="test", tag=f"{name}_216", project="e3")
        res[name] = {"taille_mo": size_mb(p), "mAP50_test": t[0], "mAP50-95_test": t[1],
                     "mAP50_216img": e[0], "mAP50-95_216img": e[1], "ap50_par_lettre": e[2],
                     "ms_par_image": bench_onnx(p, IMGSZ)}
        if name != "FP32":
            res[name]["ecart_max"], res[name]["ecart_moyen"] = score_gap(fp32, p, imgs)

    best = max(paths, key=lambda n: res[n]["mAP50_216img"])
    ref = res["FP32"]["ap50_par_lettre"]
    res["_meilleure_variante"] = best
    res["_lettres_plus_degradees"] = sorted(
        ((l, res[best]["ap50_par_lettre"].get(l, 0.0) - ap) for l, ap in ref.items()),
        key=lambda x: x[1])[:5]
    res["_calib_percentile_images"] = CALIB_PERCENTILE
    save_json("etape3.json", res)

    lines = banner("E3 — Quantization statique (CPU, 640)") + [
        f"{'':26}{'Mo':>7}{'mAP50':>8}{'216img':>8}{'ms b1':>8}{'ms b8':>8}"]
    for name in ["FP32", *paths]:
        r = res[name]
        lines.append(f"{name:26}{r['taille_mo']:>7.2f}{r['mAP50_test']:>8.3f}{r['mAP50_216img']:>8.3f}"
                     f"{r['ms_par_image'][1]:>8.1f}{r['ms_par_image'][8]:>8.1f}")
    lines.append(f"Meilleure variante : {best}")
    print("\n".join(lines))


if __name__ == "__main__":
    main()