import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

# Definition de la racine du projet et cartographie des etapes
ROOT = Path(__file__).resolve().parent
STEPS = {
    1: ("e1_onnx", "Export ONNX + PyTorch vs ONNX Runtime"),
    2: ("e2_dynamic", "Quantization dynamique"),
    3: ("e3_static", "Quantization statique (3 variantes)"),
    4: ("e4_resolution", "Resolution x quantization"),
    6: ("e6_sensibilite", "Sensibilite par bloc + calibration"),
    5: ("e5_mlflow", "MLflow + synthese + decision"),   # Execute en dernier : regroupe les resultats
}


def main():
    # 1. Parsing des arguments en ligne de commande
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", type=int, help="Numero de l'unique etape a executer")
    ap.add_argument("--from", dest="start", type=int, default=1, help="Numero de l'etape de depart")
    args = ap.parse_args()

    # 2. Verification des prérequis (presence du modele entraine best.pt)
    if not (ROOT / "model" / "best.pt").exists():
        sys.exit("Erreur: model/best.pt manquant. Veuillez d'abord executer : python training/train.py")

    # Détermination de l'ordre d'exécution des expériences
    order = [args.only] if args.only else [n for n in STEPS if n >= args.start or n == 5]
    t_all = time.perf_counter()
    
    # 3. Execution sequentielle des modules d'experience
    for n in order:
        module, label = STEPS[n]
        print(f"\n{'#' * 78}\n# Experience {n} — {label}\n{'#' * 78}")
        t0 = time.perf_counter()
        
        # Configuration de l'environnement d'execution avec PYTHONPATH
        env = {**os.environ, "PYTHONPATH": str(ROOT) + os.pathsep + os.environ.get("PYTHONPATH", "")}
        script = ROOT / "optimisation" / f"{module}.py"
        
        # Lancement du sous-processus Python
        r = subprocess.run([sys.executable, str(script)], cwd=ROOT, env=env)
        
        if r.returncode != 0:
            sys.exit(f"Erreur: Experience {n} en echec (code d'erreur {r.returncode})")
            
        print(f"OK: Experience {n} terminee en {time.perf_counter() - t0:.0f} s")
        
    # 4. Affichage du bilan final
    print(f"\nPipeline termine en {(time.perf_counter() - t_all) / 60:.1f} min — resultats enregistres dans results/")


if __name__ == "__main__":
    main()