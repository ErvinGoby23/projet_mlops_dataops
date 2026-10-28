"""
Étape 4 : corriger la boîte d'une image à la main.
Usage : python corriger_boite.py A_salon_01.jpg
- dessine le rectangle autour de la main avec la souris
- Entrée ou Espace pour valider, c pour annuler
La classe est reprise de la 1re lettre du nom du fichier.
L'aperçu (annot/apercu) est redessiné avec la nouvelle boîte en BLEU.
"""
import string
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent
IMAGES = ROOT / "annot" / "images"
LABELS = ROOT / "annot" / "labels"
APERCU = ROOT / "annot" / "apercu"
MAX_AFFICHAGE = 900

if len(sys.argv) != 2:
    raise SystemExit("Usage : python corriger_boite.py <nom_image.jpg>")

nom = sys.argv[1]
img_path = IMAGES / nom
img = cv2.imread(str(img_path))
if img is None:
    raise SystemExit(f"Image introuvable : {img_path}")

H, W = img.shape[:2]
echelle = min(1.0, MAX_AFFICHAGE / H)
affiche = cv2.resize(img, (int(W * echelle), int(H * echelle)))

x, y, w, h = cv2.selectROI("Dessine la boite puis Entree", affiche, showCrosshair=True)
cv2.destroyAllWindows()
if w == 0 or h == 0:
    raise SystemExit("Annulé, rien n'a été modifié.")

# coordonnées dans l'image réelle
x, y, w, h = x / echelle, y / echelle, w / echelle, h / echelle
cx, cy = (x + w / 2) / W, (y + h / 2) / H
cls = string.ascii_uppercase.index(nom[0].upper())   # A=0 ... Z=25

label = LABELS / f"{Path(nom).stem}.txt"
ancien = label.read_text().strip() if label.exists() else "(vide)"
nouveau = f"{cls} {cx:.6f} {cy:.6f} {w / W:.6f} {h / H:.6f}"
label.write_text(nouveau + "\n")

# aperçu redessiné : nouvelle boîte en bleu
apercu = img.copy()
cv2.rectangle(apercu, (int(x), int(y)), (int(x + w), int(y + h)), (255, 0, 0), 3)
cv2.putText(apercu, f"{nom[0].upper()} - corrigee a la main", (10, 40),
            cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 0, 0), 3)
cv2.imwrite(str(APERCU / nom), apercu)

print(f"Avant : {ancien}")
print(f"Après : {nouveau}")
print(f"Aperçu mis à jour : {APERCU / nom}")