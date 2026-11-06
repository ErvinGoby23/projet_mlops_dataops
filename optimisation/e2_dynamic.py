import sys
from pathlib import Path

# Ajout du dossier parent au chemin système pour l'importation des modules internes
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from onnxruntime.quantization import QuantType, quantize_dynamic
from optimisation.common import (
    P,
    banner,
    batch_of,
    bench_onnx,
    copy_metadata,
    evaluate,
    export_onnx,
    images_of,
    save_json,
    score_gap,
    session,
    size_mb,
    test_split,
)

# Configuration de la résolution d'image d'entrée
IMGSZ = 640


def quantize(src, dst):
    for wtype in (QuantType.QInt8, QuantType.QUInt8):
        # Applique la quantification dynamique sur le fichier source ONNX
        quantize_dynamic(str(src), str(dst), weight_type=wtype)
        try:
            # Vérifie si le modèle généré peut être chargé et exécuté par ONNX Runtime
            session(dst)
            return wtype.name
        except Exception as e:
            # Log l'échec de compatibilité du type de poids sans bloquer la boucle
            print(f"Poids {wtype.name} non executables ({str(e)[:100]}...)")
            
    # Arrêt du script si aucun type de quantification n'est pris en charge par le runtime
    raise SystemExit("Erreur: Aucune version dynamique executable.")


def main():
    # 1. Export du modèle de référence au format FP32
    fp32 = export_onnx(IMGSZ)
    dyn = P["exports"] / "asl_int8_dyn_640.onnx"
    
    # 2. Quantification dynamique et transfert des métadonnées
    wtype = quantize(fp32, dyn)
    copy_metadata(fp32, dyn)

    # 3. Évaluation de la dégradation des sorties et de la précision (mAP)
    split = test_split()
    gmax, gmean = score_gap(fp32, dyn, batch_of(images_of(split)[:16], IMGSZ))
    m32 = evaluate(fp32, IMGSZ, tag="fp32", project="e2")
    m8 = evaluate(dyn, IMGSZ, tag="int8_dyn", project="e2")
    
    # 4. Mesure des temps d'inférence (benchmarking CPU)
    s32, s8 = bench_onnx(fp32, IMGSZ), bench_onnx(dyn, IMGSZ)
    
    # Identifie les 5 classes/lettres subissant la plus forte baisse d'AP50
    deltas = sorted(
        ((l, m8[2].get(l, 0.0) - ap) for l, ap in m32[2].items()),
        key=lambda x: x[1]
    )

    # 5. Structuration et sauvegarde des résultats au format JSON
    res = {
        "split": split,
        "poids_int8": wtype,
        "onnx_fp32": {
            "taille_mo": size_mb(fp32),
            "mAP50": m32[0],
            "mAP50-95": m32[1],
            "ms_par_image": s32,
            "ap50_par_lettre": m32[2],
        },
        "onnx_int8_dynamique": {
            "taille_mo": size_mb(dyn),
            "mAP50": m8[0],
            "mAP50-95": m8[1],
            "ms_par_image": s8,
            "ap50_par_lettre": m8[2],
        },
        "ecart_sorties": {"max": gmax, "moyen": gmean},
        "lettres_plus_degradees": deltas[:5],
    }
    save_json("etape2.json", res)

    # 6. Affichage du rapport comparatif dans le terminal
    lines = banner("E2 — ONNX FP32 vs INT8 dynamique (CPU)") + [
        f"{'':20}{'FP32':>10}{'INT8 dyn':>10}",
        f"{'Taille (Mo)':20}{size_mb(fp32):>10.2f}{size_mb(dyn):>10.2f}",
        f"{'mAP50':20}{m32[0]:>10.3f}{m8[0]:>10.3f}",
        f"{'mAP50-95':20}{m32[1]:>10.3f}{m8[1]:>10.3f}",
    ]
    for bs in s32:
        lines.append(f"{'ms/img batch=' + str(bs):20}{s32[bs]:>10.1f}{s8[bs]:>10.1f}")
    lines.append(f"Poids : {wtype} | ecart scores max {gmax:.2e}")
    
    print("\n".join(lines))


if __name__ == "__main__":
    main()