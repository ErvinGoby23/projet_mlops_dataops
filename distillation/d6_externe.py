"""
D6 — Distillation sur un dataset EXTERNE (jamais vu, autre source que Roboflow).

Même test que E8 (branche main), appliqué au modèle distillé :
images rangées par lettre : <dossier>/<LETTRE>/*.jpg (pas de boîtes, la lettre = le nom du dossier).
Une image est juste si la détection la plus confiante (seuil 0,25) a la bonne lettre.
Écart B - A, IC 95 % bootstrap (10 000 tirages), test de McNemar, fidélité.
Résultat : results/d6_<tag>.json (les champs "a" et "b" indiquent les modèles comparés).

Préalable (une fois) : exporter le PROFESSEUR en ONNX FP32 416, seulement pour ce test
(il n'est jamais quantifié ni déployé) :
  python -c "from ultralytics import YOLO; from pathlib import Path; p=YOLO('model/teacher_v8s.pt').export(format='onnx', imgsz=416, dynamic=True, simplify=True, device='cpu'); Path(p).replace('exports/teacher_v8s_fp32_416.onnx')"

Commandes lancées — 1er échantillon (datasets_externe/echantillon_50, 1 300 images) :
  # distillée FP32 vs INT8 MinMax (par défaut)
  python distillation/d6_externe.py --images datasets_externe/echantillon_50 --tag mendeley_50_distill
  # distillée FP32 vs INT8 Percentile
  python distillation/d6_externe.py --images datasets_externe/echantillon_50 --b exports/distill_int8_416_percentile.onnx --tag mendeley_50_distill_percentile
  # distillée FP32 vs INT8 Percentile, en 640 et en 320
  python distillation/d6_externe.py --images datasets_externe/echantillon_50 --imgsz 640 --a exports/distill_fp32_640.onnx --b exports/distill_int8_640_percentile.onnx --tag mendeley_50_distill_640
  python distillation/d6_externe.py --images datasets_externe/echantillon_50 --imgsz 320 --a exports/distill_fp32_320.onnx --b exports/distill_int8_320_percentile.onnx --tag mendeley_50_distill_320
  # v1 vs distillée, en FP32 puis en INT8
  python distillation/d6_externe.py --images datasets_externe/echantillon_50 --a exports/asl_fp32_416.onnx --b exports/distill_fp32_416.onnx --tag mendeley_50_v1_vs_distill
  python distillation/d6_externe.py --images datasets_externe/echantillon_50 --a exports/asl_int8_static_416.onnx --b exports/distill_int8_416_percentile.onnx --tag mendeley_50_v1int8_vs_distillint8

Commandes lancées — échantillon de CONFIRMATION (datasets_externe/confirmation_50, jamais utilisé pour choisir) :
  # v1 : FP32 vs INT8 MinMax, FP32 vs INT8 Percentile, MinMax vs Percentile
  python distillation/d6_externe.py --images datasets_externe/confirmation_50 --a exports/asl_fp32_416.onnx --b exports/asl_int8_static_416_minmax.onnx --tag confirmation_v1_minmax
  python distillation/d6_externe.py --images datasets_externe/confirmation_50 --a exports/asl_fp32_416.onnx --b exports/asl_int8_static_416.onnx --tag confirmation_v1_percentile
  python distillation/d6_externe.py --images datasets_externe/confirmation_50 --a exports/asl_int8_static_416_minmax.onnx --b exports/asl_int8_static_416.onnx --tag confirmation_minmax_vs_percentile
  # distillée : FP32 vs INT8 Percentile
  python distillation/d6_externe.py --images datasets_externe/confirmation_50 --a exports/distill_fp32_416.onnx --b exports/distill_int8_416_percentile.onnx --tag confirmation_distill_percentile
  # v1 INT8 vs distillée INT8 (diapo 19 : +6,4 pts)
  python distillation/d6_externe.py --images datasets_externe/confirmation_50 --a exports/asl_int8_static_416.onnx --b exports/distill_int8_416_percentile.onnx --tag confirmation_v1int8_vs_distillint8
  # professeur FP32 vs élève FP32 (diapo 19 : 34,8 % contre 37,3 %)
  python distillation/d6_externe.py --images datasets_externe/confirmation_50 --a exports/teacher_v8s_fp32_416.onnx --b exports/distill_fp32_416.onnx --tag confirmation_prof_vs_eleve
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


def top1(model_path, files, imgsz):
    """Classe de la détection la plus confiante par image, ou None (même règle que E7 / E8)."""
    m = YOLO(str(model_path), task="detect")
    out = []
    for f in files:
        r = m.predict(str(f), imgsz=imgsz, conf=0.25, device="cpu", verbose=False)[0]
        out.append(int(r.boxes.cls[int(r.boxes.conf.argmax())]) if len(r.boxes) else None)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True, help="dossier <LETTRE>/*.jpg")
    ap.add_argument("--imgsz", type=int, default=416)
    ap.add_argument("--a", default=str(P["exports"] / "distill_fp32_416.onnx"), help="modèle A (référence)")
    ap.add_argument("--b", default=str(P["exports"] / "distill_int8_416.onnx"), help="modèle B (comparé à A)")
    ap.add_argument("--tag", default="mendeley_50_distill")
    a = ap.parse_args()

    ma, mb = Path(a.a), Path(a.b)
    for f in (ma, mb):
        if not f.exists():
            sys.exit(f"❌ {f} introuvable")
    print(f"A : {ma}\nB : {mb}")

    names = YOLO(str(ma), task="detect").names          # {0: 'A', 1: 'B', ...}
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

    pa = top1(ma, files, a.imgsz)
    pb = top1(mb, files, a.imgsz)
    oka = np.array([int(p == g) for p, g in zip(pa, gts)])
    okb = np.array([int(p == g) for p, g in zip(pb, gts)])
    d = okb - oka
    rng = np.random.default_rng(0)
    boots = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(10_000)]
    lo, hi = np.percentile(boots, [2.5, 97.5])
    n_a = int(((oka == 1) & (okb == 0)).sum())   # juste avec A seulement
    n_b = int(((oka == 0) & (okb == 1)).sum())   # juste avec B seulement
    p_mc = binomtest(min(n_a, n_b), n_a + n_b, 0.5).pvalue if n_a + n_b else 1.0
    same = float(np.mean([x == y for x, y in zip(pa, pb)]))
    rien_a = float(np.mean([p is None for p in pa]))

    perdues = Counter(names[g] for g, x, y in zip(gts, oka, okb) if x and not y)

    res = {"images": a.images, "a": str(ma), "b": str(mb), "imgsz": a.imgsz, "n": len(files),
           "exactitude_a": float(oka.mean()), "exactitude_b": float(okb.mean()),
           "ecart_b_moins_a": float(d.mean()), "ic95": [float(lo), float(hi)],
           "mcnemar_juste_a_seul": n_a, "mcnemar_juste_b_seul": n_b, "mcnemar_p": float(p_mc),
           "meme_lettre": same, "aucune_detection_a": rien_a, "perdues_par_b": dict(perdues)}
    save_json(f"d6_{a.tag}.json", res)

    print("\n".join(banner(f"D6 — A vs B sur dataset externe ({a.imgsz} px)") + [
        f"Exactitude (bonne lettre) sur {len(files)} images : A {oka.mean():.3f} | B {okb.mean():.3f}",
        f"Aucune détection (A) : {rien_a:.1%}",
        f"Écart B - A : {d.mean():+.3f}   IC 95 % [{lo:+.3f} ; {hi:+.3f}]",
        f"McNemar : {n_a} justes seulement avec A, {n_b} seulement avec B -> p = {p_mc:.2f}",
        f"Fidélité : même lettre {same:.1%}",
        f"Lettres perdues par B : {dict(perdues.most_common(8))}"]))


if __name__ == "__main__":
    main()