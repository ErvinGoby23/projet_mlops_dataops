"""
E8 — Robustesse FP32 vs INT8 sur un dataset EXTERNE (jamais vu, autre source que Roboflow).

Images rangées par lettre : <dossier>/<LETTRE>/*.jpg (pas de boîtes, la lettre = le nom du dossier).
Une image est juste si la détection la plus confiante (seuil 0,25) a la bonne lettre.
Même calcul que E7 : écart INT8 - FP32, IC 95 % bootstrap (10 000 tirages), test de McNemar, fidélité.

Usage :
  python optimisation/e8_externe.py --images datasets_externe/essai --tag essai
"""
import argparse
import sys
from collections import Counter
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
from scipy.stats import binomtest  # noqa: E402
from ultralytics import YOLO  # noqa: E402

from optimisation.common import P, banner, save_json  # noqa: E402
from optimisation.e7_robustesse import top1  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True, help="dossier <LETTRE>/*.jpg")
    ap.add_argument("--imgsz", type=int, default=416)
    ap.add_argument("--fp32")
    ap.add_argument("--int8")
    ap.add_argument("--tag", default="externe")
    a = ap.parse_args()

    fp32 = Path(a.fp32) if a.fp32 else P["exports"] / f"asl_fp32_{a.imgsz}.onnx"
    int8 = Path(a.int8) if a.int8 else P["exports"] / f"asl_int8_static_{a.imgsz}.onnx"
    for f in (fp32, int8):
        if not f.exists():
            sys.exit(f"❌ {f} introuvable")
    print(f"FP32 : {fp32}\nINT8 : {int8}")

    # Correspondance lettre -> numéro de classe, lue dans le modèle FP32
    names = YOLO(str(fp32), task="detect").names          # {0: 'A', 1: 'B', ...}
    idx = {v.upper(): k for k, v in names.items()}

    files, gts = [], []
    for d in sorted(Path(a.images).iterdir()):
        if d.is_dir() and d.name.upper() in idx:
            for f in sorted(d.glob("*.jpg")):
                files.append(f)
                gts.append(idx[d.name.upper()])
    if not files:
        sys.exit("❌ aucune image trouvée")
    print(f"{len(files)} images, {len(set(gts))} lettres")

    p32 = top1(fp32, files, a.imgsz)
    p8 = top1(int8, files, a.imgsz)
    ok32 = np.array([int(p[0] == g) for p, g in zip(p32, gts)])
    ok8 = np.array([int(p[0] == g) for p, g in zip(p8, gts)])
    d = ok8 - ok32
    rng = np.random.default_rng(0)
    boots = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(10_000)]
    lo, hi = np.percentile(boots, [2.5, 97.5])
    b = int(((ok32 == 1) & (ok8 == 0)).sum())   # juste en FP32 seulement
    c = int(((ok32 == 0) & (ok8 == 1)).sum())   # juste en INT8 seulement
    p_mc = binomtest(min(b, c), b + c, 0.5).pvalue if b + c else 1.0
    same = float(np.mean([x[0] == y[0] for x, y in zip(p32, p8)]))
    rien32 = float(np.mean([p[0] is None for p in p32]))

    # Par lettre (indicatif) : exactitude FP32 / INT8 et images perdues par l'INT8
    par_lettre = {}
    perdues = Counter()
    for g, o32, o8 in zip(gts, ok32, ok8):
        L = names[g]
        s = par_lettre.setdefault(L, [0, 0, 0])
        s[0] += 1; s[1] += int(o32); s[2] += int(o8)
        if o32 and not o8:
            perdues[L] += 1

    res = {"images": a.images, "fp32": str(fp32), "int8": str(int8), "imgsz": a.imgsz,
           "n": len(files), "exactitude_fp32": float(ok32.mean()), "exactitude_int8": float(ok8.mean()),
           "ecart_int8_moins_fp32": float(d.mean()), "ic95": [float(lo), float(hi)],
           "mcnemar_juste_fp32_seul": b, "mcnemar_juste_int8_seul": c, "mcnemar_p": float(p_mc),
           "meme_lettre": same, "aucune_detection_fp32": rien32,
           "par_lettre_n_fp32_int8": par_lettre, "perdues_par_int8": dict(perdues)}
    save_json(f"etape8_{a.tag}.json", res)

    print("\n".join(banner(f"E8 — FP32 vs INT8 sur dataset externe ({a.imgsz} px)") + [
        f"Exactitude (bonne lettre) sur {len(files)} images : FP32 {ok32.mean():.3f} | INT8 {ok8.mean():.3f}",
        f"Aucune détection (FP32) : {rien32:.1%}",
        f"Écart INT8 - FP32 : {d.mean():+.3f}   IC 95 % [{lo:+.3f} ; {hi:+.3f}]",
        f"McNemar : {b} justes seulement en FP32, {c} seulement en INT8 -> p = {p_mc:.2f}",
        f"Fidélité : même lettre {same:.1%}",
        f"Lettres perdues par l'INT8 : {dict(perdues.most_common(8))}"]))


if __name__ == "__main__":
    main()