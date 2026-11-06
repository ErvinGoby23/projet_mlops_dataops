import sys
from pathlib import Path

# Ajout du dossier parent au chemin systeme pour l'importation des modules internes
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from optimisation.common import (
    CFG,
    P,
    banner,
    bench_onnx,
    calibration_files,
    eval_yaml,
    evaluate,
    export_onnx,
    quantize_static,
    save_json,
    size_mb,
)


def main():
    # 1. Recuperation des fichiers de calibration et de la configuration des evaluations
    calib = calibration_files(CFG["quantization"]["calib_images"])
    data_eval = eval_yaml()
    models = {}
    
    # 2. Boucle d'export et de quantification INT8 pour chaque resolution (640, 416, 320)
    for size in CFG["resolutions"]:
        # Exportation du modele de base ONNX FP32 pour la resolution courante
        fp32 = export_onnx(size)
        
        # Configuration des parametres de calibration selon config.yaml
        q = CFG["quantization"]
        method = q.get("calib_method", "minmax")
        extra = {"CalibPercentile": q["calib_percentile"]} if method == "percentile" else {}
        
        # Quantification statique INT8 avec la resolution specifique
        int8 = quantize_static(
            fp32,
            P["exports"] / f"asl_int8_static_{size}.onnx",
            size,
            calib,
            method=method,
            extra_options=extra,
        )
        
        # Enregistrement des paires (chemin, resolution) pour l'evaluation
        models[f"{size}_FP32"], models[f"{size}_INT8"] = (fp32, size), (int8, size)

    # 3. Evaluation systematique de chaque variante (FP32 et INT8)
    res = {}
    for name, (path, size) in models.items():
        # Evaluation sur le split de test standard (72 images) et etendu (216 images)
        t = evaluate(path, size, tag=f"{name}_test", project="e4")
        e = evaluate(path, size, data=data_eval, split="test", tag=f"{name}_216", project="e4")
        
        # Mesure de la taille, de la precision (mAP) et de la latence (batch 1 et 8)
        res[name] = {
            "resolution": size,
            "taille_mo": size_mb(path),
            "mAP50_test": t[0],
            "mAP50-95_test": t[1],
            "mAP50_216img": e[0],
            "mAP50-95_216img": e[1],
            "ap50_par_lettre": e[2],
            "ms_par_image": bench_onnx(path, size, batches=[1, 8]),
        }

    # 4. Analyse comparative : identification des lettres les plus degradees en 320 INT8 par rapport au 640 FP32
    base_name = f"{CFG['resolutions'][0]}_FP32"
    ref = res[base_name]["ap50_par_lettre"]
    agg = res[f"{CFG['resolutions'][-1]}_INT8"]["ap50_par_lettre"]
    
    res["_lettres_plus_degradees_320_INT8"] = sorted(
        ((l, agg.get(l, 0.0) - ap) for l, ap in ref.items()),
        key=lambda x: x[1],
    )[:5]
    
    # Sauvegarde des metriques dans etape4.json
    save_json("etape4.json", res)

    # 5. Affichage du tableau de synthese des accelerations et pertes dans la console
    base = res[base_name]
    lines = banner("E4 — Resolution x quantization (CPU)") + [
        f"{'':12}{'Mo':>7}{'mAP50':>8}{'216img':>8}{'ms b1':>8}{'Accel.':>8}{'Perte':>8}"
    ]
    for name in models:
        r = res[name]
        # Calcul du facteur d'acceleration par rapport au modèle FP32 de reference
        accel = "x{:.2f}".format(base["ms_par_image"][1] / r["ms_par_image"][1])
        lines.append(
            f"{name:12}{r['taille_mo']:>7.2f}{r['mAP50_test']:>8.3f}{r['mAP50_216img']:>8.3f}"
            f"{r['ms_par_image'][1]:>8.1f}{accel:>8}{r['mAP50_216img'] - base['mAP50_216img']:>+8.3f}"
        )
    print("\n".join(lines))


if __name__ == "__main__":
    main()