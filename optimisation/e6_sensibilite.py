"""
Expérience 6 — Sensibilité de la quantization

  A. Un seul bloc du réseau quantifié à la fois -> quels blocs supportent mal l'INT8 ?
  B. Taille de la calibration (10 / 50 / 100 / 300 images) -> combien d'images faut-il ?

Résultat obtenu : 22 blocs sur 23 tolèrent l'INT8 ; seule la tête (bloc 22) s'effondre
-> justification mesurée de la précision mixte (tête en FP32).
"""
import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import onnx  # noqa: E402

from optimisation.common import (CFG, P, banner, batch_of, block_of, calibration_files,  # noqa: E402
                                 eval_yaml, evaluate, export_onnx, images_of, quantize_static,
                                 results_path, save_json, score_gap, test_split)


def main():
    s = CFG["sensibilite"]
    imgsz = s["imgsz"]
    fp32 = export_onnx(imgsz)
    tmp = P["exports"] / "sensibilite"
    tmp.mkdir(parents=True, exist_ok=True)
    charts = results_path("graphiques")
    charts.mkdir(exist_ok=True)

    blocks = {}
    for n in onnx.load(str(fp32)).graph.node:
        b = block_of(n.name)
        if b is not None:
            blocks.setdefault(b, []).append(n.name)

    calib_all = calibration_files(max(s["calib_sizes"]))
    data_eval = eval_yaml()
    imgs = batch_of(images_of(test_split())[:32], imgsz)

    ref50, ref5095, _ = evaluate(fp32, imgsz, data=data_eval, split="test", tag="fp32", project="e6")
    res = {"reference_fp32": {"mAP50": ref50, "mAP50-95": ref5095}, "blocs": {}, "calibration": {}}

    print(f"=== A. Sensibilité de {len(blocks)} blocs ===")
    for b in sorted(blocks):
        dst = quantize_static(fp32, tmp / f"bloc_{b:02d}.onnx", imgsz,
                              calib_all[:s["calib_images_blocks"]], protect_head=False,
                              nodes_to_quantize=blocks[b])
        m50, m5095, _ = evaluate(dst, imgsz, data=data_eval, split="test", tag=f"bloc_{b:02d}", project="e6")
        res["blocs"][b] = {"noeuds": len(blocks[b]), "mAP50": m50, "mAP50-95": m5095,
                           "perte_mAP50": m50 - ref50, "ecart_scores": score_gap(fp32, dst, imgs)[1]}
        print(f"  bloc {b:>2} : mAP50 {m50:.3f} ({m50 - ref50:+.3f})")

    print("=== B. Taille de la calibration ===")
    for n in s["calib_sizes"]:
        t0 = time.perf_counter()
        dst = quantize_static(fp32, tmp / f"calib_{n}.onnx", imgsz, calib_all[:n])
        dt = time.perf_counter() - t0
        m50, m5095, _ = evaluate(dst, imgsz, data=data_eval, split="test", tag=f"calib_{n}", project="e6")
        res["calibration"][n] = {"mAP50": m50, "mAP50-95": m5095, "perte_mAP50": m50 - ref50,
                                 "ecart_scores": score_gap(fp32, dst, imgs)[1], "temps_s": dt}
        print(f"  {n:>3} images : mAP50 {m50:.3f} | mAP50-95 {m5095:.3f} | {dt:.0f} s")
    save_json("etape6.json", res)

    bs = sorted(res["blocs"])
    losses = [res["blocs"][b]["perte_mAP50"] for b in bs]
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar([str(b) for b in bs], losses, color=["#e03131" if x < -0.02 else "#1c7ed6" for x in losses])
    ax.axhline(-0.02, color="#e03131", ls="--", lw=1, label="seuil −0,02")
    ax.set_xlabel("Bloc du réseau (0 = entrée … 22 = tête de détection)")
    ax.set_ylabel("Δ mAP50 vs FP32")
    ax.set_title("Sensibilité à la quantization INT8 — un seul bloc quantifié à la fois")
    ax.legend()
    fig.tight_layout()
    fig.savefig(charts / "sensibilite_blocs.png", dpi=150)

    ns = sorted(res["calibration"])
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(ns, [res["calibration"][n]["mAP50-95"] for n in ns], "o-", label="INT8 (tête FP32)")
    ax.axhline(ref5095, color="#868e96", ls="--", label="FP32")
    ax.set_xscale("log")
    ax.set_xlabel("Nombre d'images de calibration")
    ax.set_ylabel("mAP50-95 (216 images)")
    ax.set_title("Effet de la taille de la calibration")
    ax.legend()
    fig.tight_layout()
    fig.savefig(charts / "calibration.png", dpi=150)
    plt.close("all")

    worst = sorted(bs, key=lambda b: res["blocs"][b]["perte_mAP50"])[:3]
    lines = banner(f"E6 — Sensibilité ({imgsz})") + [f"Référence FP32 : mAP50 {ref50:.3f}"]
    lines += [f"  bloc {b:>2} : {res['blocs'][b]['perte_mAP50']:+.3f}" for b in worst]
    lines += [f"  calib {n:>3} : mAP50-95 {res['calibration'][n]['mAP50-95']:.3f}" for n in ns]
    print("\n".join(lines))


if __name__ == "__main__":
    main()
