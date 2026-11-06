"""
E7 — Robustesse de "pas de perte de precision" : FP32 vs INT8, meme modele, meme resolution.

1. Precision appariee sur les 216 images jamais vues (valid + test, etiquetees) :
   Chaque image est-elle bien reconnue (bonne lettre ET boite IoU > 0,5) ?
   -> Ecart INT8 - FP32, intervalle de confiance bootstrap 95 %, test de McNemar
2. Fidelite sur les 1 728 images (train + valid + test), sans etiquettes :
   L'INT8 predit-il la meme lettre que le FP32 ? (+ chevauchement des boites IoU)

Usage :
  python optimisation/e7_robustesse.py
  python optimisation/e7_robustesse.py --fp32 A.onnx --int8 B.onnx --tag distill
"""

import argparse
import sys
from pathlib import Path

# Ajout du dossier parent au chemin systeme pour l'importation des modules internes
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from scipy.stats import binomtest
from ultralytics import YOLO

from optimisation.common import P, banner, images_of, save_json


def top1(model_path, files, imgsz):
    """
    Detection la plus confiante par image : (classe, boite normalisee) ou (None, None).
    Traite une image a la fois sur CPU pour eviter la saturation memoire.
    """
    m = YOLO(str(model_path), task="detect")
    out = []
    for f in files:
        r = m.predict(str(f), imgsz=imgsz, conf=0.25, device="cpu", verbose=False)[0]
        if len(r.boxes):
            i = int(r.boxes.conf.argmax())
            out.append((int(r.boxes.cls[i]), r.boxes.xyxyn[i].tolist()))
        else:
            out.append((None, None))
    return out


def gt_of(img):
    """Recupere la vraie lettre et la vraie boite englobante depuis le fichier label YOLO."""
    lab = img.parent.parent / "labels" / f"{img.stem}.txt"
    c, cx, cy, w, h = map(float, lab.read_text().split()[:5])
    return int(c), [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2]


def iou(a, b):
    """Calcule l'Intersection over Union (IoU) entre deux boites [xmin, ymin, xmax, ymax]."""
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def correct(pred, gt):
    """Determine si la prediction est exacte (bonne classe ET IoU > 0.5 avec la verite terrain)."""
    return int(pred[0] is not None and pred[0] == gt[0] and iou(pred[1], gt[1]) > 0.5)


def main():
    # 1. Gestion des arguments de la ligne de commande
    ap = argparse.ArgumentParser()
    ap.add_argument("--imgsz", type=int, default=416)
    ap.add_argument("--fp32", help="ONNX non quantifie (defaut : celui de la config)")
    ap.add_argument("--int8", help="ONNX quantifie (defaut : celui de la config)")
    ap.add_argument("--tag", default="", help="Suffixe du fichier de resultats, ex : distill")
    a = ap.parse_args()

    # Resolution des chemins vers les modeles FP32 et INT8
    fp32 = Path(a.fp32) if a.fp32 else P["exports"] / f"asl_fp32_{a.imgsz}.onnx"
    int8 = Path(a.int8) if a.int8 else P["exports"] / f"asl_int8_static_{a.imgsz}.onnx"
    for f in (fp32, int8):
        if not f.exists():
            sys.exit(f"Erreur: {f} introuvable (lancez d'abord run_all.py ou passez --fp32/--int8)")
    print(f"FP32 : {fp32}\nINT8 : {int8}")

    # 2. Evaluation de la precision appariee sur les 216 images non vues (valid + test)
    eval_imgs = images_of("valid") + images_of("test")
    gts = [gt_of(f) for f in eval_imgs]
    
    pred_fp32_eval = top1(fp32, eval_imgs, a.imgsz)
    pred_int8_eval = top1(int8, eval_imgs, a.imgsz)
    
    ok_fp32 = np.array([correct(p, g) for p, g in zip(pred_fp32_eval, gts)])
    ok_int8 = np.array([correct(p, g) for p, g in zip(pred_int8_eval, gts)])
    
    # Calcul de l'ecart et de l'intervalle de confiance à 95% via Bootstrap (10 000 tirages)
    d = ok_int8 - ok_fp32
    rng = np.random.default_rng(0)
    boots = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(10_000)]
    lo, hi = np.percentile(boots, [2.5, 97.5])
    
    # Test statistique de McNemar (comparaison des erreurs discordantes)
    b = int(((ok_fp32 == 1) & (ok_int8 == 0)).sum())   # Juste en FP32 uniquement
    c = int(((ok_fp32 == 0) & (ok_int8 == 1)).sum())   # Juste en INT8 uniquement
    p_mcnemar = binomtest(min(b, c), b + c, 0.5).pvalue if b + c else 1.0

    # 3. Mesure de la fidelite de l'INT8 par rapport au FP32 sur l'ensemble du dataset (1 728 images)
    train_imgs = images_of("train")
    pa = top1(fp32, train_imgs, a.imgsz) + pred_fp32_eval
    pb = top1(int8, train_imgs, a.imgsz) + pred_int8_eval
    
    n_all = len(pa)
    same_all = np.mean([x[0] == y[0] for x, y in zip(pa, pb)])
    same_eval = np.mean([x[0] == y[0] for x, y in zip(pred_fp32_eval, pred_int8_eval)])
    ious = [iou(x[1], y[1]) for x, y in zip(pa, pb) if x[0] is not None and y[0] is not None]

    # 4. Sauvegarde des métriques statistiques au format JSON
    res = {
        "fp32": str(fp32),
        "int8": str(int8),
        "imgsz": a.imgsz,
        "n_eval": len(eval_imgs),
        "exactitude_fp32": float(ok_fp32.mean()),
        "exactitude_int8": float(ok_int8.mean()),
        "ecart_int8_moins_fp32": float(d.mean()),
        "ic95": [float(lo), float(hi)],
        "mcnemar_juste_fp32_seul": b,
        "mcnemar_juste_int8_seul": c,
        "mcnemar_p": float(p_mcnemar),
        "n_fidelite": n_all,
        "meme_lettre_toutes": float(same_all),
        "meme_lettre_216_non_vues": float(same_eval),
        "iou_moyen_boites": float(np.mean(ious)) if ious else None,
    }
    save_json(f"etape7{'_' + a.tag if a.tag else ''}.json", res)

    # 5. Affichage du bilan complet dans le terminal
    print("\n".join(banner(f"E7 — Robustesse FP32 vs INT8 ({a.imgsz} px)") + [
        f"Precision sur {len(eval_imgs)} images jamais vues : "
        f"FP32 {ok_fp32.mean():.3f} | INT8 {ok_int8.mean():.3f}",
        f"Ecart INT8 - FP32 : {d.mean():+.3f}   IC 95 % [{lo:+.3f} ; {hi:+.3f}]",
        f"McNemar : {b} images justes seulement en FP32, {c} seulement en INT8 -> p = {p_mcnemar:.2f}",
        f"Fidelite sur {n_all} images : meme lettre {same_all:.1%} | IoU moyen des boites {np.mean(ious):.3f}",
        f"Dont sur les {len(eval_imgs)} images jamais vues : meme lettre {same_eval:.1%}"
    ]))


if __name__ == "__main__":
    main()