"""
Expérience 2 — Quantization post-training DYNAMIQUE (quantize_dynamic, TP 7)

Question : quantifier les poids à l'avance et les activations à la volée suffit-il ?
Résultat attendu (cours) : taille ÷~4, mais vitesse pas garantie sur des convolutions.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from onnxruntime.quantization import QuantType, quantize_dynamic  # noqa: E402

from optimisation.common import (P, banner, batch_of, bench_onnx, copy_metadata, evaluate,  # noqa: E402
                                 export_onnx, images_of, save_json, score_gap, session, size_mb,
                                 test_split)

IMGSZ = 640


def quantize(src, dst):
    """Essaie int8 signé comme le TP ; ONNX Runtime CPU ne sait pas exécuter ConvInteger
    avec des poids int8 signés -> on retombe sur uint8 (constat intéressant à présenter)."""
    for wtype in (QuantType.QInt8, QuantType.QUInt8):
        quantize_dynamic(str(src), str(dst), weight_type=wtype)
        try:
            session(dst)
            return wtype.name
        except Exception as e:
            print(f"⚠️  Poids {wtype.name} non exécutables ({str(e)[:100]}…)")
    raise SystemExit("❌ Aucune version dynamique exécutable.")


def main():
    fp32 = export_onnx(IMGSZ)
    dyn = P["exports"] / "asl_int8_dyn_640.onnx"
    wtype = quantize(fp32, dyn)
    copy_metadata(fp32, dyn)

    split = test_split()
    gmax, gmean = score_gap(fp32, dyn, batch_of(images_of(split)[:16], IMGSZ))
    m32 = evaluate(fp32, IMGSZ, tag="fp32", project="e2")
    m8 = evaluate(dyn, IMGSZ, tag="int8_dyn", project="e2")
    s32, s8 = bench_onnx(fp32, IMGSZ), bench_onnx(dyn, IMGSZ)
    deltas = sorted(((l, m8[2].get(l, 0.0) - ap) for l, ap in m32[2].items()), key=lambda x: x[1])

    res = {"split": split, "poids_int8": wtype,
           "onnx_fp32": {"taille_mo": size_mb(fp32), "mAP50": m32[0], "mAP50-95": m32[1],
                         "ms_par_image": s32, "ap50_par_lettre": m32[2]},
           "onnx_int8_dynamique": {"taille_mo": size_mb(dyn), "mAP50": m8[0], "mAP50-95": m8[1],
                                   "ms_par_image": s8, "ap50_par_lettre": m8[2]},
           "ecart_sorties": {"max": gmax, "moyen": gmean},
           "lettres_plus_degradees": deltas[:5]}
    save_json("etape2.json", res)

    lines = banner("E2 — ONNX FP32 vs INT8 dynamique (CPU)") + [
        f"{'':20}{'FP32':>10}{'INT8 dyn':>10}",
        f"{'Taille (Mo)':20}{size_mb(fp32):>10.2f}{size_mb(dyn):>10.2f}",
        f"{'mAP50':20}{m32[0]:>10.3f}{m8[0]:>10.3f}",
        f"{'mAP50-95':20}{m32[1]:>10.3f}{m8[1]:>10.3f}"]
    for bs in s32:
        lines.append(f"{'ms/img batch=' + str(bs):20}{s32[bs]:>10.1f}{s8[bs]:>10.1f}")
    lines.append(f"Poids : {wtype} | écart scores max {gmax:.2e}")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
