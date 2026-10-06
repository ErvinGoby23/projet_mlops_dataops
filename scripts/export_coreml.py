"""
Export CoreML FP32 / FP16 / INT8 + comparaison (lancé par GitHub Actions sur un Mac).

- Exporte model/best.pt en 3 versions CoreML
- Mesure taille + précision (mAP sur le jeu de test) + vitesse d'inférence (Mac)
- Écrit exports/benchmark.md et exports/benchmark.csv
- Copie la version choisie (DEPLOY_VARIANT, défaut int8) dans l'app Flutter
"""
import csv
import os
import shutil
from pathlib import Path

from ultralytics import YOLO

PT = Path("model/best.pt")
OUT = Path("exports")
APP_MODEL = Path("app/assets/models/asl.mlpackage.zip")
DEPLOY = os.environ.get("DEPLOY_VARIANT", "int8")
NMS = os.environ.get("EXPORT_NMS", "true").lower() == "true"
IMGSZ = 640

VARIANTS = {
    "fp32": {},
    "fp16": {"half": True},
    "int8": {"int8": True},   # quantization post-training des poids en 8 bits
}


def size_mb(path: Path) -> float:
    if path.is_file():
        return path.stat().st_size / 1e6
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) / 1e6


def zip_mlpackage(src: Path, inner_name: str, dest_zip: Path):
    """Zip dont la racine contient <inner_name>.mlpackage (format attendu par le plugin)."""
    tmp = OUT / "_zip"
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    shutil.copytree(src, tmp / f"{inner_name}.mlpackage")
    dest_zip.parent.mkdir(parents=True, exist_ok=True)
    base = str(dest_zip).removesuffix(".zip")
    shutil.make_archive(base, "zip", root_dir=tmp, base_dir=f"{inner_name}.mlpackage")
    shutil.rmtree(tmp)


def get_dataset():
    """Télécharge le jeu de test (même dataset que l'entraînement) si la clé est fournie."""
    data = Path("datasets/asl/data.yaml")
    if data.exists():
        return str(data.resolve())
    key = os.environ.get("ROBOFLOW_API_KEY")
    if not key:
        print("⚠️  Pas de ROBOFLOW_API_KEY : export sans mesure de précision.")
        return None
    from roboflow import Roboflow
    rf = Roboflow(api_key=key)
    rf.workspace("david-lee-d0rhs").project("american-sign-language-letters") \
      .version(1).download("yolov8", location="datasets/asl")
    return str(data.resolve())


def evaluate(model_path: str, data: str | None):
    if not data:
        return None, None, None
    try:
        m = YOLO(model_path, task="detect").val(data=data, split="test", imgsz=IMGSZ,
                                                 batch=1, plots=False, verbose=False)
        return m.box.map50, m.box.map, m.speed["inference"]
    except Exception as e:
        print(f"⚠️  Évaluation impossible pour {model_path} : {e}")
        return None, None, None


def main():
    if not PT.exists():
        raise SystemExit("❌ model/best.pt introuvable : copie ton modèle entraîné dans model/")
    OUT.mkdir(exist_ok=True)
    data = get_dataset()
    rows = []

    # Référence PyTorch (non quantifié)
    map50, map5095, ms = evaluate(str(PT), data)
    rows.append({"version": "PyTorch FP32 (référence)", "taille_mo": round(size_mb(PT), 2),
                 "mAP50": map50, "mAP50-95": map5095, "inference_ms_mac": ms})

    for name, kwargs in VARIANTS.items():
        print(f"\n=== Export CoreML {name.upper()} ===")
        exported = Path(YOLO(str(PT)).export(format="coreml", imgsz=IMGSZ, nms=NMS, **kwargs))
        dest = OUT / f"asl_{name}.mlpackage"
        shutil.rmtree(dest, ignore_errors=True)
        shutil.move(str(exported), dest)
        zip_mlpackage(dest, f"asl_{name}", OUT / f"asl_{name}.mlpackage.zip")

        map50, map5095, ms = evaluate(str(dest), data)
        rows.append({"version": f"CoreML {name.upper()}", "taille_mo": round(size_mb(dest), 2),
                     "mAP50": map50, "mAP50-95": map5095, "inference_ms_mac": ms})

        if name == DEPLOY:
            zip_mlpackage(dest, "asl", APP_MODEL)
            print(f"✅ {name.upper()} copié dans l'app : {APP_MODEL}")

    # ---- Rapport ----
    fmt = lambda v, d=3: "—" if v is None else f"{v:.{d}f}"
    with open(OUT / "benchmark.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)

    md = ["## Comparaison quantization (ASL, YOLOv8n)", "",
          f"Modèle déployé dans l'app : **CoreML {DEPLOY.upper()}**", "",
          "| Version | Taille (Mo) | mAP50 | mAP50-95 | Inférence Mac (ms) |",
          "|---|---|---|---|---|"]
    for r in rows:
        md.append(f"| {r['version']} | {r['taille_mo']:.2f} | {fmt(r['mAP50'])} | "
                  f"{fmt(r['mAP50-95'])} | {fmt(r['inference_ms_mac'], 1)} |")
    md += ["", "_Vitesse mesurée sur le Mac de GitHub Actions ; la vitesse sur iPhone "
               "s'affiche en FPS dans l'app._"]
    report = "\n".join(md)
    (OUT / "benchmark.md").write_text(report)
    print("\n" + report)

    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        with open(summary, "a") as f:
            f.write(report + "\n")


if __name__ == "__main__":
    main()
