import csv
import os
import shutil
from pathlib import Path

from ultralytics import YOLO

PT = Path("model/best.pt")
OUT = Path("exports")
APP_MODELS = Path("app/assets/models")
DATA = Path("datasets/asl/data.yaml")
NMS = os.environ.get("EXPORT_NMS", "true").lower() == "true"

VARIANTS = {
    "fp32_640": (640, {}),
    "int8_640": (640, {"int8": True}),
    "fp16_416": (416, {"half": True}),
    "int8_416": (416, {"int8": True}),
    "int8_320": (320, {"int8": True}),
}


# Donne la taille d'un fichier ou d'un dossier en Mo
def size_mb(path: Path) -> float:
    if path.is_file():
        return path.stat().st_size / 1e6
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) / 1e6


# Met le modèle .mlpackage dans un fichier .zip pour l'application
def zip_mlpackage(src: Path, inner_name: str, dest_zip: Path):
    tmp = OUT / "_zip"
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    shutil.copytree(src, tmp / f"{inner_name}.mlpackage")
    dest_zip.parent.mkdir(parents=True, exist_ok=True)
    base = str(dest_zip).removesuffix(".zip")
    shutil.make_archive(base, "zip", root_dir=tmp, base_dir=f"{inner_name}.mlpackage")
    shutil.rmtree(tmp)


# Trouve le dataset sur l'ordinateur ou le télécharge depuis Roboflow
def get_dataset():
    if DATA.exists():
        return str(DATA.resolve())
    key = os.environ.get("ROBOFLOW_API_KEY")
    if not key:
        print("Attention: Pas de ROBOFLOW_API_KEY declaree. Export sans mesure de precision.")
        return None
    from roboflow import Roboflow
    rf = Roboflow(api_key=key)
    project = rf.workspace("david-lee-d0rhs").project("american-sign-language-letters")
    project.version(1).download("yolov8", location="datasets/asl")
    return str(DATA.resolve())


# Mesure la précision et la vitesse du modèle sur les images de test
def evaluate(model_path: str, data, imgsz: int):
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


# Exporte une version du modèle en CoreML, la zippe et renvoie ses résultats
def export_variant(name: str, imgsz: int, kwargs: dict, data):
    print(f"\n=== Export CoreML {name} ===")
    exported = Path(YOLO(str(PT)).export(format="coreml", imgsz=imgsz, nms=NMS, **kwargs))
    dest = OUT / f"asl_{name}.mlpackage"
    shutil.rmtree(dest, ignore_errors=True)
    shutil.move(str(exported), dest)
    zip_mlpackage(dest, f"asl_{name}", APP_MODELS / f"asl_{name}.mlpackage.zip")

    map50, map5095, ms = evaluate(str(dest), data, imgsz)
    return {
        "version": f"CoreML {name}",
        "resolution": imgsz,
        "taille_mo": round(size_mb(dest), 2),
        "mAP50": map50,
        "mAP50-95": map5095,
        "inference_ms_mac": ms,
    }


# Affiche un nombre avec des décimales, ou un tiret s'il n'y a pas de valeur
def fmt(value, decimals=3):
    return "—" if value is None else f"{value:.{decimals}f}"


# Enregistre les résultats dans un fichier CSV
def save_csv(rows):
    with open(OUT / "benchmark.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)


# Crée le tableau des résultats au format Markdown
def build_report(rows):
    md = [
        "## CoreML — versions embarquees dans l'app", "",
        "| Version | Resolution | Taille (Mo) | mAP50 | mAP50-95 | Inference Mac (ms) |",
        "|---|---|---|---|---|---|",
    ]
    for r in rows:
        md.append(
            f"| {r['version']} | {r['resolution']} | {r['taille_mo']:.2f} | {fmt(r['mAP50'])} | "
            f"{fmt(r['mAP50-95'])} | {fmt(r['inference_ms_mac'], 1)} |"
        )
    md += ["", "_Les FPS en stream se mesurent dans l'app sur l'iPhone (bouton « Mesurer 10 s »)._"]
    return "\n".join(md)


# Ajoute le rapport au résumé de GitHub Actions s'il existe
def publish_summary(report):
    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        with open(summary, "a", encoding="utf-8") as f:
            f.write(report + "\n")


# Lance tout : export des modèles, mesures, puis création des rapports
def main():
    if not PT.exists():
        raise SystemExit("Erreur: model/best.pt introuvable")

    OUT.mkdir(exist_ok=True)
    for old in APP_MODELS.glob("*.zip"):
        old.unlink()

    data = get_dataset()
    rows = [export_variant(name, imgsz, kwargs, data)
            for name, (imgsz, kwargs) in VARIANTS.items()]

    save_csv(rows)
    report = build_report(rows)
    (OUT / "benchmark.md").write_text(report, encoding="utf-8")
    print("\n" + report)
    publish_summary(report)


if __name__ == "__main__":
    main()