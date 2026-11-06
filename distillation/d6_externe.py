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
            sys.exit(f"{f} introuvable")
    print(f"A : {ma}\nB : {mb}")

    names = YOLO(str(ma), task="detect").names
    idx = {v.upper(): k for k, v in names.items()}

    files, gts = [], []
    for d in sorted(Path(a.images).iterdir()):
        if d.is_dir() and d.name.upper() in idx:
            for f in sorted(d.glob("*.jpg")):
                files.append(f)
                gts.append(idx[d.name.upper()])
    if not files:
        sys.exit("aucune image trouvée")
    print(f"{len(files)} images, {len(set(gts))} lettres")

    pa = top1(ma, files, a.imgsz)
    pb = top1(mb, files, a.imgsz)
    oka = np.array([int(p == g) for p, g in zip(pa, gts)])
    okb = np.array([int(p == g) for p, g in zip(pb, gts)])
    d = okb - oka
    rng = np.random.default_rng(0)
    boots = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(10_000)]
    lo, hi = np.percentile(boots, [2.5, 97.5])
    n_a = int(((oka == 1) & (okb == 0)).sum())
    n_b = int(((oka == 0) & (okb == 1)).sum())
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