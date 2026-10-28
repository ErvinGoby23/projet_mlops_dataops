"""
Étape 5 : séparation entraînement / test réel de mes photos.
Règle : *_01.jpg -> train ; *_02.jpg -> test_reel (même lieu : test un peu optimiste).
Sortie : v2/train/{images,labels}, v2/test_reel/{images,labels}
         data_test_reel.yaml (mesurer un modèle sur mon test réel)
         data_v2.yaml        (entraîner la v2 : dataset David Lee + mon train)
"""
from pathlib import Path
import shutil

import yaml

ROOT = Path(__file__).resolve().parent              # datasets_ervin
REPO = ROOT.parent                                  # racine du repo
ANNOT = ROOT / "annot"
OUT = ROOT / "v2"
DATASET = REPO / "datasets" / "asl"                 # dataset David Lee


def est_test(nom):
    return Path(nom).stem.endswith("_02")


def main():
    # noms des classes : on reprend EXACTEMENT ceux du dataset d'origine
    names = yaml.safe_load((DATASET / "data.yaml").read_text())["names"]

    if OUT.exists():
        shutil.rmtree(OUT)                          # on repart propre à chaque lancement
    for split in ("train", "test_reel"):
        for sous in ("images", "labels"):
            (OUT / split / sous).mkdir(parents=True)

    compte, vides = {"train": 0, "test_reel": 0}, []
    for img in sorted((ANNOT / "images").glob("*.jpg")):
        label = ANNOT / "labels" / f"{img.stem}.txt"
        if not label.exists() or not label.read_text().strip():
            vides.append(img.name)                  # pas d'annotation : on ne l'utilise pas
            continue
        split = "test_reel" if est_test(img.name) else "train"
        shutil.copy2(img, OUT / split / "images" / img.name)
        shutil.copy2(label, OUT / split / "labels" / label.name)
        compte[split] += 1

    # 1) pour mesurer un modèle sur MON test réel
    test_yaml = {
        "path": OUT.as_posix(),
        "train": "test_reel/images",   # obligatoire pour YOLO, non utilisé pour l'évaluation
        "val": "test_reel/images",
        "test": "test_reel/images",
        "names": names,
    }
    (ROOT / "data_test_reel.yaml").write_text(yaml.safe_dump(test_yaml, allow_unicode=True))

    # 2) pour entraîner la v2 : dataset David Lee + mon train
    v2_yaml = {
        "train": [(DATASET / "train" / "images").as_posix(),
                  (OUT / "train" / "images").as_posix()],
        "val": (DATASET / "valid" / "images").as_posix(),
        "test": (DATASET / "test" / "images").as_posix(),
        "names": names,
    }
    (ROOT / "data_v2.yaml").write_text(yaml.safe_dump(v2_yaml, allow_unicode=True))

    print(f"train     : {compte['train']} photos")
    print(f"test_reel : {compte['test_reel']} photos")
    if vides:
        print("Ignorées (sans annotation) :", *vides, sep="\n  ")
    print("\nÉcrit : v2/, data_test_reel.yaml, data_v2.yaml")
    print("Classes :", names if isinstance(names, list) else list(names.values()))


if __name__ == "__main__":
    main()