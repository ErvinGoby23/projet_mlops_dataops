

import csv
import os
import shutil
from pathlib import Path

from ultralytics import YOLO

# Definition des chemins principaux et variables d'environnement
PT = Path("model/best_distill.pt")
OUT = Path("exports")
APP_MODELS = Path("app/assets/models")
NMS = os.environ.get("EXPORT_NMS", "true").lower() == "true"

# Dictionnaire des variantes : Nom - (Resolution, Options d'export)
VARIANTS = {
    "fp32_640": (640, {}),
    "int8_640": (640, {"int8": True}),
    "fp16_416": (416, {"half": True}),
    "int8_416": (416, {"int8": True}),
    "int8_320": (320, {"int8": True}),
}


def size_mb(path: Path) -> float:
    """Calcule la taille d'un fichier ou d'un repertoire (.mlpackage) en Mo."""
    if path.is_file():
        return path.stat().st_size / 1e6
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) / 1e6


def zip_mlpackage(src: Path, inner_name: str, dest_zip: Path):
    """Compresse le dossier .mlpackage en un archive .zip lisible par l'application Flutter."""
    tmp = OUT / "_zip"
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    shutil.copytree(src, tmp / f"{inner_name}.mlpackage")
    dest_zip.parent.mkdir(parents=True, exist_ok=True)
    base = str(dest_zip).removesuffix(".zip")
    shutil.make_archive(base, "zip", root_dir=tmp, base_dir=f"{inner_name}.mlpackage")
    shutil.rmtree(tmp)


def get_dataset():
    """Verifie la presence du dataset local ou le telecharge via l'API Roboflow."""
    data = Path("datasets/asl/data.yaml")
    if data.exists():
        return str(data.resolve())
    key = os.environ.get("ROBOFLOW_API_KEY")
    if not key:
        print("Attention: Pas de ROBOFLOW_API_KEY declaree. Export sans mesure de precision.")
        return None
    from roboflow import Roboflow
    rf = Roboflow(api_key=key)
    rf.workspace("david-lee-d0rhs").project("american-sign-language-letters") \
      .version(1).download("yolov8", location="datasets/asl")
    return str(data.resolve())


def evaluate(model_path: str, data, imgsz: int):
    """Evalue le modele exporte avec YOLO.val() sur le jeu de test."""
    if not data:
        return None, None, None
    try:
        m = YOLO(model_path, task="detect").val(
            data=data, split="test", imgsz=imgsz, batch=1, plots=False, verbose=False
        )
        return m.box.map50, m.box.map, m.speed["inference"]
    except Exception as e:
        print(f"Attention: Evaluation impossible pour {model_path} ({e})")
        return None, None, None


def main():
    if not PT.exists():
        raise SystemExit("Erreur: model/best_distill.pt introuvable")
        
    OUT.mkdir(exist_ok=True)
    
    # Nettoyage des anciennes archives de modeles dans le dossier Flutter
    for old in APP_MODELS.glob("*.zip"):
        old.unlink()
        
    data = get_dataset()
    rows = []

    # Exportation et evaluation de chaque variante CoreML
    for name, (imgsz, kwargs) in VARIANTS.items():
        print(f"\n=== Export CoreML {name} ===")
        exported = Path(YOLO(str(PT)).export(format="coreml", imgsz=imgsz, nms=NMS, **kwargs))
        dest = OUT / f"asl_{name}.mlpackage"
        shutil.rmtree(dest, ignore_errors=True)
        shutil.move(str(exported), dest)
        zip_mlpackage(dest, f"asl_{name}", APP_MODELS / f"asl_{name}.mlpackage.zip")

        map50, map5095, ms = evaluate(str(dest), data, imgsz)
        rows.append({
            "version": f"CoreML {name}",
            "resolution": imgsz,
            "taille_mo": round(size_mb(dest), 2),
            "mAP50": map50,
            "mAP50-95": map5095,
            "inference_ms_mac": ms
        })

    fmt = lambda v, d=3: "—" if v is None else f"{v:.{d}f}"
    
    # Sauvegarde des resultats dans un fichier CSV
    with open(OUT / "benchmark.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)

    # Generation du rapport au format Markdown
    md = [
        "## CoreML — versions embarquees dans l'app", "",
        "| Version | Resolution | Taille (Mo) | mAP50 | mAP50-95 | Inference Mac (ms) |",
        "|---|---|---|---|---|---|"
    ]
    for r in rows:
        md.append(
            f"| {r['version']} | {r['resolution']} | {r['taille_mo']:.2f} | {fmt(r['mAP50'])} | "
            f"{fmt(r['mAP50-95'])} | {fmt(r['inference_ms_mac'], 1)} |"
        )
    md += ["", "_Les FPS en stream se mesurent dans l'app sur l'iPhone (bouton « Mesurer 10 s »)._"]
    
    report = "\n".join(md)
    (OUT / "benchmark.md").write_text(report)
    print("\n" + report)
    
    # Publication du rapport dans le sommaire du job GitHub Actions si disponible
    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        with open(summary, "a") as f:
            f.write(report + "\n")


if __name__ == "__main__":
    main()