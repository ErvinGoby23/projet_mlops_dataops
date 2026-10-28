"""
Étape 6 : mesure du data drift de la v1.
Évalue model/best.pt sur le test du dataset (72 images) et sur mon test réel,
en 640 et en 416. Résultats : results/v2_etape6_drift.json
"""
from pathlib import Path
import json

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent        # datasets_ervin
REPO = ROOT.parent
MODEL = REPO / "model" / "best.pt"
JEUX = {
    "test_dataset": REPO / "datasets" / "asl" / "data.yaml",
    "test_reel": ROOT / "data_test_reel.yaml",
}
RESOLUTIONS = [640, 416]


def main():
    model = YOLO(str(MODEL))
    resultats = {}
    for jeu, data in JEUX.items():
        for imgsz in RESOLUTIONS:
            m = model.val(data=str(data), split="test", imgsz=imgsz, batch=1,
                          device="cpu", plots=False, verbose=False,
                          project=str(REPO / "runs" / "v2_drift"),
                          name=f"{jeu}_{imgsz}", exist_ok=True)
            resultats[f"{jeu}_{imgsz}"] = {
                "mAP50": round(float(m.box.map50), 3),
                "mAP50-95": round(float(m.box.map), 3),
                "P": round(float(m.box.mp), 3),
                "R": round(float(m.box.mr), 3),
            }

    (REPO / "results").mkdir(exist_ok=True)
    out = REPO / "results" / "v2_etape6_drift.json"
    out.write_text(json.dumps(resultats, indent=2))

    print(f"\n{'jeu':<20}{'mAP50':>8}{'mAP50-95':>10}{'P':>8}{'R':>8}")
    for k, v in resultats.items():
        print(f"{k:<20}{v['mAP50']:>8}{v['mAP50-95']:>10}{v['P']:>8}{v['R']:>8}")
    for imgsz in RESOLUTIONS:
        d = resultats[f"test_dataset_{imgsz}"]["mAP50"] - resultats[f"test_reel_{imgsz}"]["mAP50"]
        print(f"Écart de mAP50 dataset -> réel en {imgsz} : {d:+.3f}")
    print(f"\nÉcrit : {out}")


if __name__ == "__main__":
    main()