"""
Copie tous les HEIC/HEIF d'un dossier vers un autre dossier.
Usage : python scripts/copier_heif.py <dossier_entree> <dossier_sortie>
"""
import shutil
import sys
from pathlib import Path

if len(sys.argv) != 3:
    raise SystemExit("Usage : python scripts/copier_heif.py <dossier_entree> <dossier_sortie>")

src = Path(sys.argv[1])
dst = Path(sys.argv[2])
dst.mkdir(parents=True, exist_ok=True)

fichiers = sorted(f for f in src.iterdir() if f.suffix.lower() in {".heic", ".heif"})

for f in fichiers:
    shutil.copy2(f, dst / f.name)  # copy2 garde la date de prise de vue

print(f"{len(fichiers)} fichiers HEIC/HEIF copiés dans {dst}")