"""
Point d'entrée unique du pipeline de quantization.

    python run_all.py              # toutes les expériences (1 -> 6, puis 5 pour la synthèse)
    python run_all.py --only 3     # une seule expérience
    python run_all.py --from 4     # à partir de l'expérience 4

Chaque expérience écrit ses résultats dans results/etapeN.json ;
l'expérience 5 (MLflow) relit tout et produit la synthèse + le choix du modèle.
"""
import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
STEPS = {
    1: ("e1_onnx", "Export ONNX + PyTorch vs ONNX Runtime"),
    2: ("e2_dynamic", "Quantization dynamique"),
    3: ("e3_static", "Quantization statique (3 variantes)"),
    4: ("e4_resolution", "Résolution × quantization"),
    6: ("e6_sensibilite", "Sensibilité par bloc + calibration"),
    5: ("e5_mlflow", "MLflow + synthèse + décision"),      # en dernier : relit tout
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", type=int)
    ap.add_argument("--from", dest="start", type=int, default=1)
    args = ap.parse_args()

    if not (ROOT / "model" / "best.pt").exists():
        sys.exit("❌ model/best.pt manquant : lance d'abord  python training/train.py")

    order = [args.only] if args.only else [n for n in STEPS if n >= args.start or n == 5]
    t_all = time.perf_counter()
    for n in order:
        module, label = STEPS[n]
        print(f"\n{'#' * 78}\n# Expérience {n} — {label}\n{'#' * 78}")
        t0 = time.perf_counter()
        env = {**os.environ, "PYTHONPATH": str(ROOT) + os.pathsep + os.environ.get("PYTHONPATH", "")}
        script = ROOT / "optimisation" / f"{module}.py"
        r = subprocess.run([sys.executable, str(script)], cwd=ROOT, env=env)
        if r.returncode != 0:
            sys.exit(f"❌ Expérience {n} en échec (code {r.returncode})")
        print(f"✅ Expérience {n} terminée en {time.perf_counter() - t0:.0f} s")
    print(f"\n🏁 Pipeline terminé en {(time.perf_counter() - t_all) / 60:.1f} min — résultats dans results/")


if __name__ == "__main__":
    main()
