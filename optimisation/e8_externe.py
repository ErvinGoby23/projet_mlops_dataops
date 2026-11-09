import argparse
import sys
from collections import Counter
from pathlib import Path

# Ajout du dossier parent au chemin systeme pour l'importation des modules internes
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from scipy.stats import binomtest
from ultralytics import YOLO

from optimisation.common import P, banner, save_json
from optimisation.e7_robustesse import top1


def main():
    # 1. Parsing des arguments en ligne de commande
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True, help="Dossier contenant la structure <LETTRE>/*.jpg")
    ap.add_argument("--imgsz", type=int, default=416)
    ap.add_argument("--fp32", help="Chemin du modele ONNX FP32 (optionnel)")
    ap.add_argument("--int8", help="Chemin du modele ONNX INT8 (optionnel)")
    ap.add_argument("--tag", default="externe", help="Tag pour le fichier de resultats JSON")
    a = ap.parse_args()

    # Resolution des chemins vers les modeles
    fp32 = Path(a.fp32) if a.fp32 else P["exports"] / f"asl_fp32_{a.imgsz}.onnx"
    int8 = Path(a.int8) if a.int8 else P["exports"] / f"asl_int8_static_{a.imgsz}.onnx"
    
    for f in (fp32, int8):
        if not f.exists():
            sys.exit(f"Erreur: {f} introuvable")
    print(f"FP32 : {fp32}\nINT8 : {int8}")

    # 2. Mapping entre le nom de la lettre et son identifiant de classe
    names = YOLO(str(fp32), task="detect").names
    idx = {v.upper(): k for k, v in names.items()}

    # Parcours des dossiers de lettres et collecte des chemins d'images et labels (gts)
    files, gts = [], []
    for d in sorted(Path(a.images).iterdir()):
        if d.is_dir() and d.name.upper() in idx:
            for f in sorted(d.glob("*.jpg")):
                files.append(f)
                gts.append(idx[d.name.upper()])
                
    if not files:
        sys.exit("Erreur: Aucune image trouvee dans le repertoire specifie")
    print(f"{len(files)} images chargees sur {len(set(gts))} lettres distinctes")

    # 3. Inferences et calculs des predictions top-1 pour FP32 et INT8
    p32 = top1(fp32, files, a.imgsz)
    p8 = top1(int8, files, a.imgsz)
    
    ok32 = np.array([int(p[0] == g) for p, g in zip(p32, gts)])
    ok8 = np.array([int(p[0] == g) for p, g in zip(p8, gts)])
    
    # Estimation bootstrap de l'ecart moyen (IC a 95%)
    d = ok8 - ok32
    rng = np.random.default_rng(0)
    boots = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(10_000)]
    lo, hi = np.percentile(boots, [2.5, 97.5])
    
    # Test statistique de McNemar
    b = int(((ok32 == 1) & (ok8 == 0)).sum())   # Juste uniquement en FP32
    c = int(((ok32 == 0) & (ok8 == 1)).sum())   # Juste uniquement en INT8
    p_mc = binomtest(min(b, c), b + c, 0.5).pvalue if b + c else 1.0
    
    # Fidelite globale et taux d'absence de detection
    same = float(np.mean([x[0] == y[0] for x, y in zip(p32, p8)]))
    rien32 = float(np.mean([p[0] is None for p in p32]))

    # 4. Analyse detaillee par lettre et suivi des degradations INT8
    par_lettre = {}
    perdues = Counter()
    for g, o32, o8 in zip(gts, ok32, ok8):
        L = names[g]
        s = par_lettre.setdefault(L, [0, 0, 0])
        s[0] += 1
        s[1] += int(o32)
        s[2] += int(o8)
        if o32 and not o8:
            perdues[L] += 1

    # 5. Export des metriques au format JSON
    res = {
        "images": a.images,
        "fp32": str(fp32),
        "int8": str(int8),
        "imgsz": a.imgsz,
        "n": len(files),
        "exactitude_fp32": float(ok32.mean()),
        "exactitude_int8": float(ok8.mean()),
        "ecart_int8_moins_fp32": float(d.mean()),
        "ic95": [float(lo), float(hi)],
        "mcnemar_juste_fp32_seul": b,
        "mcnemar_juste_int8_seul": c,
        "mcnemar_p": float(p_mc),
        "meme_lettre": same,
        "aucune_detection_fp32": rien32,
        "par_lettre_n_fp32_int8": par_lettre,
        "perdues_par_int8": dict(perdues),
    }
    save_json(f"etape8_{a.tag}.json", res)

    # 6. Affichage du bilan dans le terminal
    print("\n".join(banner(f"E8 — FP32 vs INT8 sur dataset externe ({a.imgsz} px)") + [
        f"Exactitude (bonne lettre) sur {len(files)} images : FP32 {ok32.mean():.3f} | INT8 {ok8.mean():.3f}",
        f"Aucune detection (FP32) : {rien32:.1%}",
        f"Ecart INT8 - FP32 : {d.mean():+.3f}   IC 95 % [{lo:+.3f} ; {hi:+.3f}]",
        f"McNemar : {b} justes seulement en FP32, {c} seulement en INT8 -> p = {p_mc:.2f}",
        f"Fidelite : meme lettre {same:.1%}",
        f"Lettres perdues par l'INT8 : {dict(perdues.most_common(8))}"
    ]))


if __name__ == "__main__":
    main()