"""
Démo PC (plan B de l'oral) : webcam + modèles ONNX, même logique que l'app iPhone.
Le modèle n'analyse que le CARRÉ au centre (gros plan, comme les photos d'entraînement).
    python demo/demo_pc.py
Touches : m = modèle suivant | i = inverser (miroir) | espace | b = effacer | c = tout effacer | q = quitter
"""
from pathlib import Path
import re
import time

import cv2
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
NOMS = [
    "asl_int8_static_416.onnx",   # retenu
    "asl_fp32_416.onnx",
    "asl_int8_static_640.onnx",
    "asl_fp32_640.onnx",
    "asl_int8_static_320.onnx",
    "asl_fp32_320.onnx",
]
MODELES = [ROOT / "exports" / n for n in NOMS if (ROOT / "exports" / n).exists()]
SEUIL = 0.55        # même seuil que l'app
MAINTIEN_MS = 900   # une lettre doit être tenue 900 ms pour être validée
ZONE = 0.75         # taille du carré = 75 % de la hauteur de l'image
CAMERA = 0


def imgsz_de(chemin):
    m = re.search(r"(640|416|320)", chemin.name)
    return int(m.group(1)) if m else 640


def charger(i):
    return YOLO(str(MODELES[i]), task="detect"), imgsz_de(MODELES[i])


def main():
    if not MODELES:
        raise SystemExit("Aucun modèle trouvé dans exports/")
    cap = cv2.VideoCapture(CAMERA, cv2.CAP_DSHOW)
    if not cap.isOpened():
        raise SystemExit("Webcam introuvable")

    i = 0
    model, imgsz = charger(i)
    miroir = False
    mot, candidat, depuis, temps = "", None, 0.0, []

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if miroir:
            frame = cv2.flip(frame, 1)

        # carré centré : seule cette zone est envoyée au modèle
        H, W = frame.shape[:2]
        c = int(H * ZONE)
        x0, y0 = (W - c) // 2, (H - c) // 2
        zone = frame[y0:y0 + c, x0:x0 + c]

        t0 = time.perf_counter()
        r = model.predict(zone, imgsz=imgsz, conf=SEUIL, device="cpu", verbose=False)[0]
        ms = (time.perf_counter() - t0) * 1000
        temps = (temps + [ms])[-30:]

        cv2.rectangle(frame, (x0, y0), (x0 + c, y0 + c), (255, 200, 0), 2)  # le carré guide

        lettre = None
        if len(r.boxes):
            k = int(r.boxes.conf.argmax())
            lettre = model.names[int(r.boxes.cls[k])]
            conf = float(r.boxes.conf[k])
            bx1, by1, bx2, by2 = map(int, r.boxes.xyxy[k].tolist())
            cv2.rectangle(frame, (x0 + bx1, y0 + by1), (x0 + bx2, y0 + by2), (0, 200, 0), 3)
            cv2.putText(frame, f"{lettre} {conf:.2f}", (x0 + bx1, max(30, y0 + by1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 200, 0), 2)

        maintenant = time.time()
        if lettre != candidat:
            candidat, depuis = lettre, maintenant
        elif lettre and (maintenant - depuis) * 1000 >= MAINTIEN_MS:
            if not mot or mot[-1] != lettre:
                mot += lettre
            depuis = float("inf")

        moy = sum(temps) / len(temps)
        infos = [f"Modele : {MODELES[i].name} ({imgsz})",
                 f"Inference : {moy:.1f} ms  (~{1000 / moy:.0f} FPS)",
                 f"Miroir : {'oui' if miroir else 'non'}   Mot : {mot}"]
        for j, txt in enumerate(infos):
            cv2.putText(frame, txt, (10, 30 + 30 * j), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (255, 255, 255), 2)
        cv2.imshow("ASL - demo PC (main dans le carre)", frame)

        t = cv2.waitKey(1) & 0xFF
        if t == ord("q"):
            break
        if t == ord("m"):
            i = (i + 1) % len(MODELES)
            model, imgsz = charger(i)
            temps = []
        elif t == ord("i"):
            miroir = not miroir
        elif t == ord(" "):
            mot += " "
        elif t == ord("b"):
            mot = mot[:-1]
        elif t == ord("c"):
            mot = ""

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()