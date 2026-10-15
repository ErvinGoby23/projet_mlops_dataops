"""
ÉTAPE 1 — Export ONNX + comparaison PyTorch vs ONNX Runtime (comme le TP 7, partie 1)

Mesure, pour le modèle original (PyTorch) et sa version ONNX :
  - la taille du fichier
  - la précision (mAP50, mAP50-95) sur le jeu de test
  - la vitesse d'inférence sur CPU pour batch = 1, 8, 32 (warm-up + moyenne)
  - l'écart entre les sorties brutes des deux modèles (soft-predictions)

Tout tourne sur CPU : c'est plus proche d'un appareil embarqué qu'une RTX 5060.

Lancement (dans le dossier "projet", venv activé) :
    pip install onnx onnxruntime onnxslim
    python .\\etape1_onnx.py
"""
import json
import time
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
import torch
from ultralytics import YOLO

MODEL_PT = Path(r"model\best.pt")
DATA = Path(r"datasets\asl\data.yaml")
OUT = Path("exports")
RESULTS = Path("results")
IMGSZ = 640
BATCHES = [1, 8, 32]
N_RUNS_BS1 = 50          # nombre de passages pour batch=1 (moins pour les gros batchs)


# ---------------------------------------------------------------------------
def letterbox(path, size):
    """Même prétraitement qu'Ultralytics : redimensionne sans déformer + bandes grises."""
    img = cv2.imread(str(path))
    h, w = img.shape[:2]
    r = min(size / h, size / w)
    nh, nw = round(h * r), round(w * r)
    img = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    top, left = (size - nh) // 2, (size - nw) // 2
    canvas = np.full((size, size, 3), 114, dtype=np.uint8)
    canvas[top:top + nh, left:left + nw] = img
    return canvas[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) / 255.0   # BGR->RGB, HWC->CHW


def test_images(n):
    root = DATA.parent
    split = "test" if (root / "test" / "images").exists() else "valid"
    files = sorted((root / split / "images").glob("*"))[:n]
    return np.stack([letterbox(f, IMGSZ) for f in files]), split


def size_mb(p):
    return Path(p).stat().st_size / 1e6


def evaluate(model_path, split):
    m = YOLO(str(model_path), task="detect").val(
        data=str(DATA), split=split, imgsz=IMGSZ, batch=1, device="cpu",
        workers=0, plots=False, verbose=False, project="runs/etape1", name="val", exist_ok=True)
    return float(m.box.map50), float(m.box.map)


def bench(run_fn, bs):
    """Warm-up (le 1er appel est plus lent) puis moyenne sur plusieurs passages."""
    n = max(5, N_RUNS_BS1 // bs)
    x = np.random.rand(bs, 3, IMGSZ, IMGSZ).astype(np.float32)
    for _ in range(3):
        run_fn(x)
    t0 = time.perf_counter()
    for _ in range(n):
        run_fn(x)
    ms_batch = (time.perf_counter() - t0) / n * 1000
    return ms_batch, ms_batch / bs


# ---------------------------------------------------------------------------
def main():
    OUT.mkdir(exist_ok=True)
    RESULTS.mkdir(exist_ok=True)
    print(f"CPU : {torch.get_num_threads()} threads PyTorch")

    # 1. Export ONNX (batch dynamique pour pouvoir tester 1 / 8 / 32)
    onnx_path = OUT / "asl_fp32_640.onnx"
    exported = YOLO(str(MODEL_PT)).export(format="onnx", imgsz=IMGSZ, dynamic=True,
                                          simplify=True, device="cpu")
    Path(exported).replace(onnx_path)
    print(f"\n✅ ONNX exporté : {onnx_path}")

    # 2. Modèles prêts à l'emploi
    net = YOLO(str(MODEL_PT)).model.float().eval().cpu()
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    inp = sess.get_inputs()[0].name

    def run_torch(x):
        with torch.inference_mode():
            return net(torch.from_numpy(x))[0].numpy()

    def run_onnx(x):
        return sess.run(None, {inp: x})[0]

    # 3. Écart des sorties sur de vraies images (soft-predictions)
    imgs, split = test_images(16)
    out_t, out_o = run_torch(imgs), run_onnx(imgs)
    diff_scores = np.abs(out_t[:, 4:, :] - out_o[:, 4:, :])    # lignes 4+ = scores des 26 lettres
    max_diff, mean_diff = float(diff_scores.max()), float(diff_scores.mean())

    # 4. Précision
    print("\nÉvaluation de la précision (jeu de test)…")
    map_t = evaluate(MODEL_PT, split)
    map_o = evaluate(onnx_path, split)

    # 5. Vitesse
    print("\nMesure de la vitesse sur CPU…")
    speed = {"pytorch": {}, "onnx": {}}
    for bs in BATCHES:
        speed["pytorch"][bs] = bench(run_torch, bs)
        speed["onnx"][bs] = bench(run_onnx, bs)
        print(f"  batch {bs:>2} : PyTorch {speed['pytorch'][bs][1]:.1f} ms/img | "
              f"ONNX {speed['onnx'][bs][1]:.1f} ms/img")

    # 6. Rapport
    res = {
        "split": split,
        "pytorch": {"taille_mo": size_mb(MODEL_PT), "mAP50": map_t[0], "mAP50-95": map_t[1],
                    "ms_par_image": {bs: v[1] for bs, v in speed["pytorch"].items()}},
        "onnx_fp32": {"taille_mo": size_mb(onnx_path), "mAP50": map_o[0], "mAP50-95": map_o[1],
                      "ms_par_image": {bs: v[1] for bs, v in speed["onnx"].items()}},
        "ecart_sorties": {"max": max_diff, "moyen": mean_diff},
    }
    (RESULTS / "etape1.json").write_text(json.dumps(res, indent=2))

    p, o = res["pytorch"], res["onnx_fp32"]
    lines = [
        "", "=" * 64, "RÉSULTATS ÉTAPE 1 — PyTorch vs ONNX Runtime (CPU)", "=" * 64,
        f"{'':18}{'PyTorch':>12}{'ONNX FP32':>12}",
        f"{'Taille (Mo)':18}{p['taille_mo']:>12.2f}{o['taille_mo']:>12.2f}",
        f"{'mAP50':18}{p['mAP50']:>12.3f}{o['mAP50']:>12.3f}",
        f"{'mAP50-95':18}{p['mAP50-95']:>12.3f}{o['mAP50-95']:>12.3f}",
    ]
    for bs in BATCHES:
        lines.append(f"{f'ms/img batch={bs}':18}{p['ms_par_image'][bs]:>12.1f}{o['ms_par_image'][bs]:>12.1f}")
    lines += [
        f"Écart des scores PyTorch/ONNX : max {max_diff:.2e} | moyen {mean_diff:.2e}",
        f"(évalué sur le split '{split}')", "=" * 64,
        f"Résultats sauvegardés dans {RESULTS / 'etape1.json'}",
    ]
    print("\n".join(lines))


if __name__ == "__main__":
    main()
