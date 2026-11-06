import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from optimisation.common import CFG, P, ROOT  # noqa: E402
from optimisation.e5_mlflow import dataset_fingerprint  # noqa: E402

RESULTS = P["results"]
ONNX_INT8 = ROOT / "exports" / "d9_distill_s0_int8_416_percentile.onnx"
REGISTERED_NAME = CFG["mlflow"]["registered_model"]
ALIAS = "production"


def main():
    import mlflow
    import mlflow.onnx
    import onnx
    from mlflow.tracking import MlflowClient

    if not ONNX_INT8.exists():
        raise SystemExit(f"{ONNX_INT8} introuvable")

    uri = CFG["mlflow"]["tracking_uri"]
    if uri.startswith("sqlite:///") and not Path(uri[10:]).is_absolute():
        uri = "sqlite:///" + str((ROOT / uri[10:]).as_posix())
    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment(CFG["mlflow"]["experiment"])

    fp, counts = dataset_fingerprint()

    params = {
        "format": "ONNX", "precision": "INT8", "imgsz": 416,
        "methode": "PTQ statique Percentile, tête FP32",
        "entrainement": "distillation, professeur YOLOv8s, graine 0",
        "dataset_train": counts["train"], "dataset_valid": counts["valid"], "dataset_test": counts["test"],
    }
    metrics = {
        "taille_mo": ONNX_INT8.stat().st_size / 1e6,
        "externe_fp32": 0.373,
        "externe_int8": 0.342,
        "externe_gain_vs_v1_int8": 0.064,
        "externe_vs_v8m_int8_p": 0.77,
        "graines_int8_moyenne": 0.317,
        "graines_int8_ecart_type": 0.018,
        "graines_gain_vs_controle": 0.072,
        "coreml_int8_416_mAP50": 0.874,
        "coreml_int8_416_mAP50-95": 0.709,
        "coreml_int8_416_taille_mo": 3.20,
    }

    with mlflow.start_run(run_name="distillee_int8_416_percentile") as run:
        mlflow.set_tags({"etape": "d10", "dataset_empreinte": fp, "choisi": "production"})
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)

        for f in sorted(RESULTS.glob("d6_confirmation_*.json")) + [RESULTS / "d9_graines.json"]:
            if f.exists():
                mlflow.log_artifact(str(f), artifact_path="resultats")
        if (RESULTS / "graphiques").exists():
            mlflow.log_artifacts(str(RESULTS / "graphiques"), artifact_path="graphiques")
        mlflow.log_artifact(str(ROOT / "config.yaml"))

        model = onnx.load(str(ONNX_INT8))
        try:
            info = mlflow.onnx.log_model(model, name="model", registered_model_name=REGISTERED_NAME)
        except TypeError:
            info = mlflow.onnx.log_model(model, artifact_path="model", registered_model_name=REGISTERED_NAME)

    client = MlflowClient()
    version = getattr(info, "registered_model_version", None) or max(
        int(m.version) for m in client.search_model_versions(f"name='{REGISTERED_NAME}'"))
    client.set_registered_model_alias(REGISTERED_NAME, ALIAS, str(version))

    print(f"Run {run.info.run_id}")
    print(f"models:/{REGISTERED_NAME}@{ALIAS} (v{version}) — l'alias « production » pointe maintenant sur la distillée")


if __name__ == "__main__":
    main()