

import sys
import time
from pathlib import Path

# Ajout du dossier parent au chemin systeme pour les importations internes
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from optimisation.common import (
    CFG,
    P,
    banner,
    batch_of,
    bench_onnx,
    calibration_files,
    eval_yaml,
    evaluate,
    export_onnx,
    images_of,
    quantize_static,
    save_json,
    score_gap,
    size_mb,
    test_split,
)

# Resolution d'image d'entree pour l'experience
IMGSZ = 640

# Dictionnaire des variantes de quantification statique a tester
VARIANTS = {
    "A_minmax_tout": dict(method="minmax", protect_head=False, extra={}),
    "B_minmax_tete_fp32": dict(method="minmax", protect_head=True, extra={}),
    "C_percentile_tete_fp32": dict(
        method="percentile", protect_head=True, extra={"CalibPercentile": 99.999}
    ),
}


def main():
    # 1. Exportation ou chargement du modele de reference ONNX en FP32
    fp32 = export_onnx(IMGSZ)
    
    # 2. Preparation de la liste des images de calibration
    calib = calibration_files(CFG["quantization"]["calib_images"])
    paths = {}
    
    # 3. Execution de la quantification statique pour chaque variante
    for name, v in VARIANTS.items():
        t0 = time.perf_counter()
        
        # Generation du fichier ONNX quantifie INT8 avec les options specifiques
        paths[name] = quantize_static(
            fp32,
            P["exports"] / f"asl_int8_static_{name}.onnx",
            IMGSZ,
            calib,
            method=v["method"],
            protect_head=v["protect_head"],
            extra_options=v["extra"],
        )
        print(f"OK: {name} ({size_mb(paths[name]):.2f} Mo, {time.perf_counter() - t0:.0f} s)")

    # 4. Evaluation des performances sur le jeu de donnees
    data_eval = eval_yaml()
    imgs = batch_of(images_of(test_split())[:16], IMGSZ)
    res = {}
    
    # Mesure du mAP, de la taille et de la latence pour le FP32 et chaque variante INT8
    for name, p in {"FP32": fp32, **paths}.items():
        t = evaluate(p, IMGSZ, tag=f"{name}_test", project="e3")
        e = evaluate(p, IMGSZ, data=data_eval, split="test", tag=f"{name}_216", project="e3")
        
        res[name] = {
            "taille_mo": size_mb(p),
            "mAP50_test": t[0],
            "mAP50-95_test": t[1],
            "mAP50_216img": e[0],
            "mAP50-95_216img": e[1],
            "ap50_par_lettre": e[2],
            "ms_par_image": bench_onnx(p, IMGSZ),
        }
        
        # Calcul de l'ecart moyen et maximal par rapport au modele FP32
        if name != "FP32":
            res[name]["ecart_max"], res[name]["ecart_moyen"] = score_gap(fp32, p, imgs)

    # 5. Identification de la meilleure variante sur 216 images et calcul des pertes par lettre
    best = max(paths, key=lambda n: res[n]["mAP50_216img"])
    ref = res["FP32"]["ap50_par_lettre"]
    res["_meilleure_variante"] = best
    res["_lettres_plus_degradees"] = sorted(
        ((l, res[best]["ap50_par_lettre"].get(l, 0.0) - ap) for l, ap in ref.items()),
        key=lambda x: x[1],
    )[:5]
    
    # Sauvegarde des metriques completes au format JSON
    save_json("etape3.json", res)

    # 6. Affichage du tableau recapitulatif dans la console
    lines = banner("E3 — Quantization statique (CPU, 640)") + [
        f"{'':26}{'Mo':>7}{'mAP50':>8}{'216img':>8}{'ms b1':>8}{'ms b8':>8}"
    ]
    for name in ["FP32", *paths]:
        r = res[name]
        lines.append(
            f"{name:26}{r['taille_mo']:>7.2f}{r['mAP50_test']:>8.3f}{r['mAP50_216img']:>8.3f}"
            f"{r['ms_par_image'][1]:>8.1f}{r['ms_par_image'][8]:>8.1f}"
        )
    lines.append(f"Meilleure variante : {best}")
    print("\n".join(lines))


if __name__ == "__main__":
    main()