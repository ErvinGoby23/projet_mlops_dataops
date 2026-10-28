"""
Étape 3 : pré-annotation de mes photos avec le modèle v1.
Entrée : jpg/<LETTRE>/*.jpg
Sortie : annot/images/*.jpg + annot/labels/*.txt (format YOLO)
         annot/apercu/*.jpg  (boîte dessinée, pour vérifier à l'œil)
         annot/rapport.csv   (lettre attendue, lettre prédite par la v1, confiance)
La CLASSE vient du dossier ; seule la BOÎTE vient du modèle.
Les images déjà annotées (et peut-être corrigées à la main) ne sont PAS retraitées.
"""
from pathlib import Path
import csv
import shutil

import cv2
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent               # datasets_ervin
MODEL = ROOT.parent / "model" / "best.pt"            # modèle v1
SRC = ROOT / "jpg"
OUT = ROOT / "annot"
CONF_MIN = 0.05   # seuil très bas : on veut une boîte même si la v1 hésite
IMGSZ = 640       # résolution d'entraînement de best.pt


def main():
    model = YOLO(str(MODEL))
    nom_vers_id = {nom: i for i, nom in model.names.items()}   # 'A' -> 0 ...

    for sous in ("images", "labels", "apercu"):
        (OUT / sous).mkdir(parents=True, exist_ok=True)

    lignes, sans_boite, deja = [], [], 0
    for lettre_dir in sorted(p for p in SRC.iterdir() if p.is_dir()):
        lettre = lettre_dir.name.upper()
        cls = nom_vers_id[lettre]
        for img_path in sorted(lettre_dir.glob("*.jpg")):
            label_path = OUT / "labels" / f"{img_path.stem}.txt"
            if label_path.exists():
                deja += 1
                continue   # déjà annotée (et peut-être corrigée) : on n'y touche pas

            r = model.predict(str(img_path), imgsz=IMGSZ, conf=CONF_MIN,
                              device="cpu", verbose=False)[0]
            shutil.copy2(img_path, OUT / "images" / img_path.name)

            if len(r.boxes) == 0:
                label_path.write_text("")          # à annoter à la main
                sans_boite.append(img_path.name)
                lignes.append([img_path.name, lettre, "", "", "AUCUNE BOITE"])
                continue

            best = int(r.boxes.conf.argmax())      # boîte la plus sûre
            x, y, w, h = r.boxes.xywhn[best].tolist()
            conf = float(r.boxes.conf[best])
            pred = model.names[int(r.boxes.cls[best])]

            # classe imposée par le dossier, boîte donnée par le modèle
            label_path.write_text(f"{cls} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")

            # aperçu : boîte + lettre attendue / prédite
            img = cv2.imread(str(img_path))
            x1, y1, x2, y2 = map(int, r.boxes.xyxy[best].tolist())
            ok = pred == lettre
            couleur = (0, 200, 0) if ok else (0, 0, 255)
            cv2.rectangle(img, (x1, y1), (x2, y2), couleur, 3)
            cv2.putText(img, f"attendu {lettre} / v1 {pred} {conf:.2f}",
                        (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.1, couleur, 3)
            cv2.imwrite(str(OUT / "apercu" / img_path.name), img)

            lignes.append([img_path.name, lettre, pred, f"{conf:.3f}",
                           "OK" if ok else "ERREUR"])

    # le rapport est COMPLÉTÉ (pas écrasé) à chaque lancement
    rapport = OUT / "rapport.csv"
    nouveau = not rapport.exists()
    with open(rapport, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f, delimiter=";")
        if nouveau:
            w.writerow(["image", "attendu", "predit_v1", "confiance", "statut"])
        w.writerows(lignes)

    n = len(lignes)
    print(f"Déjà annotées (ignorées) : {deja}")
    print(f"Nouvelles images traitées : {n}")
    if n:
        n_ok = sum(1 for l in lignes if l[4] == "OK")
        print(f"Boîte trouvée : {n - len(sans_boite)} / {n}")
        print(f"Lettre correcte selon la v1 : {n_ok} / {n} ({100 * n_ok / n:.0f} %)")
    if sans_boite:
        print("Sans boîte (à annoter à la main) :", *sans_boite, sep="\n  ")
    print(f"\nRésultats dans {OUT} (voir apercu/ et rapport.csv)")


if __name__ == "__main__":
    main()