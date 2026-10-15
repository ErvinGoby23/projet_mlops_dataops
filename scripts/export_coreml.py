"""
Export CoreML de plusieurs versions + évaluation (lancé par GitHub Actions sur un Mac).

Versions embarquées dans l'app (l'utilisateur choisit dans l'app, et mesure les FPS en stream) :
  fp32_640 | int8_640 | fp16_416 | int8_416 | int8_320

Note : en CoreML, "int8" = quantization des POIDS sur 8 bits (le calcul se fait ensuite
en FP16 sur le Neural Engine). Il n'y a pas de quantization des activations, donc pas
le problème de la tête de détection vu avec ONNX à l'étape 3.
"""
import csv
import os
import shutil
from pathlib import Path

from ultralytics import YOLO

PT = Path("model/best.pt")
OUT = Path("exports")
APP_MODELS = Path("app/assets/models")
NMS = os.environ.get("EXPORT_NMS", "true").lower() == "true"

# nom -> (résolution, options d'export)
VARIANTS = {
    "fp32_640": (640, {}),
    "int8_640": (640, {"int8": True}),
    "fp16_416": (416, {"half": True}),
    "int8_416": (416, {"int8": True}),
    "int8_320": (320, {"int8": True}),
}


def size_mb(path: Path) -> float:
    if path.is_file():
        return path.stat().st_size / 1e6
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) / 1e6


def zip_mlpackage(src: Path, inner_name: str, dest_zip: Path):
    tmp = OUT / "_zip"
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    shutil.copytree(src, tmp / f"{inner_name}.mlpackage")
    dest_zip.parent.mkdir(parents=True, exist_ok=True)
    base = str(dest_zip).removesuffix(".zip")
    shutil.make_archive(base, "zip", root_dir=tmp, base_dir=f"{inner_name}.mlpackage")
    shutil.rmtree(tmp)


def get_dataset():
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


def evaluate(model_path: str, data, imgsz: int):
    if not data:
        return None, None, None
    try:
        m = YOLO(model_path, task="detect").val(data=data, split="test", imgsz=imgsz,
                                                 batch=1, plots=False, verbose=False)
        return m.box.map50, m.box.map, m.speed["inference"]
    except Exception as e:
        print(f"⚠️  Évaluation impossible pour {model_path} : {e}")
        return None, None, None


def main():
    if not PT.exists():
        raise SystemExit("❌ model/best.pt introuvable")
    OUT.mkdir(exist_ok=True)
    for old in APP_MODELS.glob("*.zip"):
        old.unlink()
    data = get_dataset()
    rows = []

    for name, (imgsz, kwargs) in VARIANTS.items():
        print(f"\n=== Export CoreML {name} ===")
        exported = Path(YOLO(str(PT)).export(format="coreml", imgsz=imgsz, nms=NMS, **kwargs))
        dest = OUT / f"asl_{name}.mlpackage"
        shutil.rmtree(dest, ignore_errors=True)
        shutil.move(str(exported), dest)
        zip_mlpackage(dest, f"asl_{name}", APP_MODELS / f"asl_{name}.mlpackage.zip")

        map50, map5095, ms = evaluate(str(dest), data, imgsz)
        rows.append({"version": f"CoreML {name}", "resolution": imgsz,
                     "taille_mo": round(size_mb(dest), 2), "mAP50": map50,
                     "mAP50-95": map5095, "inference_ms_mac": ms})

    fmt = lambda v, d=3: "—" if v is None else f"{v:.{d}f}"
    with open(OUT / "benchmark.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)

    md = ["## CoreML — versions embarquées dans l'app", "",
          "| Version | Résolution | Taille (Mo) | mAP50 | mAP50-95 | Inférence Mac (ms) |",
          "|---|---|---|---|---|---|"]
    for r in rows:
        md.append(f"| {r['version']} | {r['resolution']} | {r['taille_mo']:.2f} | {fmt(r['mAP50'])} | "
                  f"{fmt(r['mAP50-95'])} | {fmt(r['inference_ms_mac'], 1)} |")
    md += ["", "_Les FPS en stream se mesurent dans l'app sur l'iPhone (bouton « Mesurer 10 s »)._"]
    report = "\n".join(md)
    (OUT / "benchmark.md").write_text(report)
    print("\n" + report)
    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        with open(summary, "a") as f:
            f.write(report + "\n")


if __name__ == "__main__":
    main()
