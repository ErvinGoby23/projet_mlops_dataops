"""
Experience 6 — Sensibilite de la quantization

A. Un seul bloc du reseau quantifie a la fois -> identification des blocs sensibles a l'INT8
B. Taille de la calibration (10 / 50 / 100 / 300 images) -> determination du besoin minimal en donnees

Resultat attendu : 22 blocs sur 23 tolerent l'INT8 ; seule la tete (bloc 22) subit une forte degradation.
Cela apporte la justification experimentale de la precision mixte (conservation de la tete en FP32).
"""

import sys
import time
from pathlib import Path

# Ajout du dossier parent au chemin systeme pour l'importation des modules internes
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use("Agg")  # Configuration de Matplotlib en mode non interactif
import matplotlib.pyplot as plt
import onnx

from optimisation.common import (
    CFG,
    P,
    banner,
    batch_of,
    block_of,
    calibration_files,
    eval_yaml,
    evaluate,
    export_onnx,
    images_of,
    quantize_static,
    results_path,
    save_json,
    score_gap,
    test_split,
)


def main():
    # 1. Preparation des dossiers de travail et chargement du modele de reference ONNX FP32
    s = CFG["sensibilite"]
    imgsz = s["imgsz"]
    fp32 = export_onnx(imgsz)
    
    tmp = P["exports"] / "sensibilite"
    tmp.mkdir(parents=True, exist_ok=True)
    
    charts = results_path("graphiques")
    charts.mkdir(exist_ok=True)

    # 2. Regroupement des noeuds du graphe ONNX par bloc fonctionnel
    blocks = {}
    for n in onnx.load(str(fp32)).graph.node:
        b = block_of(n.name)
        if b is not None:
            blocks.setdefault(b, []).append(n.name)

    # Preparation du dataset de calibration et de l'echantillon de test
    calib_all = calibration_files(max(s["calib_sizes"]))
    data_eval = eval_yaml()
    imgs = batch_of(images_of(test_split())[:32], imgsz)

    # Evaluation du modele FP32 de reference
    ref50, ref5095, _ = evaluate(fp32, imgsz, data=data_eval, split="test", tag="fp32", project="e6")
    res = {"reference_fp32": {"mAP50": ref50, "mAP50-95": ref5095}, "blocs": {}, "calibration": {}}

    # 3. PARTIE A : Sensibilite bloc par bloc (quantification isolee)
    print(f"=== A. Sensibilite de {len(blocks)} blocs ===")
    for b in sorted(blocks):
        # Quantification d'un unique bloc tout en conservant le reste du reseau en FP32
        dst = quantize_static(
            fp32,
            tmp / f"bloc_{b:02d}.onnx",
            imgsz,
            calib_all[:s["calib_images_blocks"]],
            protect_head=False,
            nodes_to_quantize=blocks[b],
        )
        
        # Mesure de la perte de mAP50 et de l'ecart des scores par rapport au FP32
        m50, m5095, _ = evaluate(dst, imgsz, data=data_eval, split="test", tag=f"bloc_{b:02d}", project="e6")
        res["blocs"][b] = {
            "noeuds": len(blocks[b]),
            "mAP50": m50,
            "mAP50-95": m5095,
            "perte_mAP50": m50 - ref50,
            "ecart_scores": score_gap(fp32, dst, imgs)[1],
        }
        print(f"  bloc {b:>2} : mAP50 {m50:.3f} ({m50 - ref50:+.3f})")

    # 4. PARTIE B : Impact du volume de donnees de calibration
    print("=== B. Taille de la calibration ===")
    for n in s["calib_sizes"]:
        t0 = time.perf_counter()
        
        # Quantification de l'ensemble du réseau (avec tete protetee) selon le nombre d'images N
        dst = quantize_static(fp32, tmp / f"calib_{n}.onnx", imgsz, calib_all[:n])
        dt = time.perf_counter() - t0
        
        m50, m5095, _ = evaluate(dst, imgsz, data=data_eval, split="test", tag=f"calib_{n}", project="e6")
        res["calibration"][n] = {
            "mAP50": m50,
            "mAP50-95": m5095,
            "perte_mAP50": m50 - ref50,
            "ecart_scores": score_gap(fp32, dst, imgs)[1],
            "temps_s": dt,
        }
        print(f"  {n:>3} images : mAP50 {m50:.3f} | mAP50-95 {m5095:.3f} | {dt:.0f} s")
        
    # Sauvegarde des résultats au format JSON
    save_json("etape6.json", res)

    # 5. Generation du graphique : Perte de mAP50 par bloc
    bs = sorted(res["blocs"])
    losses = [res["blocs"][b]["perte_mAP50"] for b in bs]
    
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar([str(b) for b in bs], losses, color=["#e03131" if x < -0.02 else "#1c7ed6" for x in losses])
    ax.axhline(-0.02, color="#e03131", ls="--", lw=1, label="seuil -0,02")
    ax.set_xlabel("Bloc du reseau (0 = entree ... 22 = tete de detection)")
    ax.set_ylabel("Delta mAP50 vs FP32")
    ax.set_title("Sensibilite a la quantization INT8 — un seul bloc quantifie a la fois")
    ax.legend()
    fig.tight_layout()
    fig.savefig(charts / "sensibilite_blocs.png", dpi=150)

    # 6. Generation du graphique : Evolution des performances selon la calibration
    ns = sorted(res["calibration"])
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(ns, [res["calibration"][n]["mAP50-95"] for n in ns], "o-", label="INT8 (tete FP32)")
    ax.axhline(ref5095, color="#868e96", ls="--", label="FP32")
    ax.set_xscale("log")
    ax.set_xlabel("Nombre d'images de calibration")
    ax.set_ylabel("mAP50-95 (216 images)")
    ax.set_title("Effet de la taille de la calibration")
    ax.legend()
    fig.tight_layout()
    fig.savefig(charts / "calibration.png", dpi=150)
    plt.close("all")

    # 7. Affichage du compte-rendu dans le terminal
    worst = sorted(bs, key=lambda b: res["blocs"][b]["perte_mAP50"])[:3]
    lines = banner(f"E6 — Sensibilite ({imgsz})") + [f"Reference FP32 : mAP50 {ref50:.3f}"]
    lines += [f"  bloc {b:>2} : {res['blocs'][b]['perte_mAP50']:+.3f}" for b in worst]
    lines += [f"  calib {n:>3} : mAP50-95 {res['calibration'][n]['mAP50-95']:.3f}" for n in ns]
    print("\n".join(lines))


if __name__ == "__main__":
    main()