"""
Étape 8 : comparaison v1 / v2 (et v1_bis si présent) sur les deux tests, en 640 et 416.
Résultats : results/v2_etape8_comparaison.json
"""
from pathlib import Path
import json

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent
MODELES = {
    "v1": REPO / "model" / "best.pt",
    "v1_bis": REPO / "model" / "best_v1_bis.pt",   # ré-entraînement de la v1 (mesure du hasard)
    "v2": REPO / "model" / "best_v2.pt",
}
JEUX = {
    "test_dataset": REPO / "datasets" / "asl" / "data.yaml",
    "test_reel": ROOT / "data_test_reel.yaml",
}
RESOLUTIONS = [640, 416]


def main():
    res = {}
    for nom, chemin in MODELES.items():
        if not chemin.exists():
            print(f"(absent, ignoré : {chemin.name})")
            continue
        model = YOLO(str(chemin))
        for jeu, data in JEUX.items():
            for imgsz in RESOLUTIONS:
                m = model.val(data=str(data), split="test", imgsz=imgsz, batch=1,
                              device="cpu", plots=False, verbose=False,
                              project=str(REPO / "runs" / "v2_comparaison"),
                              name=f"{nom}_{jeu}_{imgsz}", exist_ok=True)
                res[f"{nom}|{jeu}|{imgsz}"] = {"mAP50": round(float(m.box.map50), 3),
                                               "mAP50-95": round(float(m.box.map), 3)}

    out = REPO / "results" / "v2_etape8_comparaison.json"
    out.write_text(json.dumps(res, indent=2))

    print(f"\n{'modèle':<8}{'jeu':<14}{'imgsz':>6}{'mAP50':>8}{'mAP50-95':>10}")
    for k, v in res.items():
        nom, jeu, imgsz = k.split("|")
        print(f"{nom:<8}{jeu:<14}{imgsz:>6}{v['mAP50']:>8}{v['mAP50-95']:>10}")
    print(f"\nÉcrit : {out}")


if __name__ == "__main__":
    main()