"""
Prépare les photos iPhone pour le dataset v2.
Entrée  : brutes/<lieu>/<LETTRE>/*.heic|jpg|png
Sortie  : jpg/<LETTRE>/<LETTRE>_<lieu>_<NN>.jpg
- convertit HEIC -> JPEG
- applique la rotation de l'iPhone (EXIF)
- redimensionne (plus grand côté = 1024 px)
- supprime les métadonnées (dont la position GPS)
"""
from pathlib import Path
import string

from PIL import Image, ImageOps
from pillow_heif import register_heif_opener

register_heif_opener()

ROOT = Path(__file__).resolve().parent   # = datasets_ervin
SRC = ROOT / "brutes"
DST = ROOT / "jpg"
MAX_SIDE = 1024
EXTS = {".heic", ".heif", ".jpg", ".jpeg", ".png"}
LETTRES = set(string.ascii_uppercase)


def fichiers_uniques(dossier):
    """Une seule version par photo : si IMG_1.HEIC et IMG_1.jpg existent, on garde le HEIC."""
    par_nom = {}
    for f in sorted(dossier.iterdir()):
        if f.suffix.lower() not in EXTS:
            continue
        deja = par_nom.get(f.stem)
        if deja is None or f.suffix.lower() in {".heic", ".heif"}:
            par_nom[f.stem] = f
    return sorted(par_nom.values())


def main():
    if not SRC.exists():
        raise SystemExit(f"Dossier introuvable : {SRC}")

    total, ignores = 0, []
    for lieu_dir in sorted(p for p in SRC.iterdir() if p.is_dir()):
        lieu = lieu_dir.name.lower().replace(" ", "-")
        for lettre_dir in sorted(p for p in lieu_dir.iterdir() if p.is_dir()):
            lettre = lettre_dir.name.upper()
            if lettre not in LETTRES:
                ignores.append(str(lettre_dir))
                continue
            out_dir = DST / lettre
            out_dir.mkdir(parents=True, exist_ok=True)

            for i, f in enumerate(fichiers_uniques(lettre_dir), start=1):
                img = Image.open(f)
                img = ImageOps.exif_transpose(img)   # rotation correcte
                img = img.convert("RGB")
                img.thumbnail((MAX_SIDE, MAX_SIDE))  # garde les proportions
                out = out_dir / f"{lettre}_{lieu}_{i:02d}.jpg"
                img.save(out, "JPEG", quality=90)    # sans EXIF -> pas de GPS
                total += 1

    print(f"{total} photos écrites dans {DST}")
    if ignores:
        print("Dossiers ignorés (pas une lettre A-Z) :", *ignores, sep="\n  ")

    print("\nComptage par lettre :")
    for lettre in sorted(LETTRES):
        d = DST / lettre
        n = len(list(d.glob("*.jpg"))) if d.exists() else 0
        print(f"  {lettre}: {n}")


if __name__ == "__main__":
    main()