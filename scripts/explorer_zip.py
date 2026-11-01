# Affiche la structure d'un zip sans l'extraire : dossiers, nombre d'images, exemples.
import sys, zipfile, collections

zf = zipfile.ZipFile(sys.argv[1])
noms = [n for n in zf.namelist() if not n.endswith("/")]
print("Fichiers :", len(noms))

# nombre de fichiers par dossier (tous les niveaux sauf le nom du fichier)
dossiers = collections.Counter("/".join(n.split("/")[:-1]) for n in noms)
print("Dossiers :", len(dossiers))
for d, c in sorted(dossiers.items())[:40]:
    print(f"{c:7d}  {d}")

print("\nExemples :")
for n in noms[:10]:
    print(" ", n, zf.getinfo(n).file_size, "octets")