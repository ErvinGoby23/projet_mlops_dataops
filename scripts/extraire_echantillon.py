# Extrait un échantillon aléatoire (graine fixe) d'images d'un type, pour les 26 lettres.
# Usage : python scripts/extraire_echantillon.py <Root.zip> <dossier_sortie> <n_par_lettre> [type]
import sys, zipfile, random, collections, pathlib

zip_path, sortie, n = sys.argv[1], pathlib.Path(sys.argv[2]), int(sys.argv[3])
type_ = sys.argv[4] if len(sys.argv) > 4 else "Type_01_(Raw_Gesture)"
LETTRES = [chr(c) for c in range(ord("A"), ord("Z") + 1)]   # DEL et SPACE exclus

zf = zipfile.ZipFile(zip_path)
noms = [x for x in zf.namelist() if not x.endswith("/")]
print("Types :", sorted({x.split("/")[1] for x in noms}))

par_lettre = collections.defaultdict(list)
for x in noms:
    parts = x.split("/")
    if parts[1] == type_ and parts[2] in LETTRES:
        par_lettre[parts[2]].append(x)

random.seed(0)
total = 0
for l in LETTRES:
    choix = random.sample(sorted(par_lettre[l]), n)
    (sortie / l).mkdir(parents=True, exist_ok=True)
    for x in choix:
        (sortie / l / x.split("/")[-1]).write_bytes(zf.read(x))
    total += len(choix)
print(f"{total} images extraites dans {sortie}")