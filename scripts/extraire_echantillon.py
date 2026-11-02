# Extrait un échantillon aléatoire (graine fixe) d'images d'un type, pour les 26 lettres.
# Option --exclure : dossier d'un échantillon précédent, dont les images ne seront pas reprises.
# Usage : python scripts/extraire_echantillon.py <Root.zip> <sortie> <n_par_lettre> [--graine 0] [--exclure dossier]
import argparse, zipfile, random, collections, pathlib

ap = argparse.ArgumentParser()
ap.add_argument("zip")
ap.add_argument("sortie")
ap.add_argument("n", type=int)
ap.add_argument("--type", default="Type_01_(Raw_Gesture)")
ap.add_argument("--graine", type=int, default=0)
ap.add_argument("--exclure", help="dossier d'un échantillon déjà utilisé")
a = ap.parse_args()

LETTRES = [chr(c) for c in range(ord("A"), ord("Z") + 1)]   # DEL et SPACE exclus
deja = set()
if a.exclure:
    deja = {(f.parent.name, f.name) for f in pathlib.Path(a.exclure).rglob("*.jpg")}
    print(f"{len(deja)} images déjà utilisées, exclues")

zf = zipfile.ZipFile(a.zip)
par_lettre = collections.defaultdict(list)
for x in zf.namelist():
    p = x.split("/")
    if len(p) == 4 and p[1] == a.type and p[2] in LETTRES and p[3] and (p[2], p[3]) not in deja:
        par_lettre[p[2]].append(x)

random.seed(a.graine)
sortie, total = pathlib.Path(a.sortie), 0
for l in LETTRES:
    choix = random.sample(sorted(par_lettre[l]), a.n)
    (sortie / l).mkdir(parents=True, exist_ok=True)
    for x in choix:
        (sortie / l / x.split("/")[-1]).write_bytes(zf.read(x))
    total += len(choix)
print(f"{total} images extraites dans {sortie}")