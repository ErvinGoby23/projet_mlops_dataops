"""
ÉTAPE 2 — Quantization post-training DYNAMIQUE (comme le TP 7, partie 2)

Principe : les POIDS sont convertis en int8 à l'avance ; les ACTIVATIONS sont
quantifiées à la volée pendant l'inférence. Pas besoin de données de calibration.

Compare ONNX FP32 (référence de l'étape 1) et ONNX INT8 dynamique :
  - taille, mAP50 / mAP50-95, vitesse CPU (batch 1 / 8 / 32)
  - écart des sorties (soft-predictions)
  - précision LETTRE PAR LETTRE (quelles lettres souffrent de la quantization ?)

Lancement :  python .\\etape2_dynamic.py
(il faut avoir lancé l'étape 1 avant : exports\\asl_fp32_640.onnx)
"""
import json
import time
from pathlib import Path

import cv2
import numpy as np
import onnx
import onnxruntime as ort
from onnxruntime.quantization import QuantType, quantize_dynamic
from ultralytics import YOLO

DATA = Path(r"datasets\asl\data.yaml")
FP32 = Path(r"exports\asl_fp32_640.onnx")
OUT = Path("exports")
RESULTS = Path("results")
IMGSZ = 640
BATCHES = [1, 8, 32]
N_RUNS_BS1 = 50


# ---------------------------------------------------------------------------
def letterbox(path, size):
    img = cv2.imread(str(path))
    h, w = img.shape[:2]
    r = min(size / h, size / w)
    nh, nw = round(h * r), round(w * r)
    img = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    top, left = (size - nh) // 2, (size - nw) // 2
    canvas = np.full((size, size, 3), 114, dtype=np.uint8)
    canvas[top:top + nh, left:left + nw] = img
    return canvas[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) / 255.0


def test_images(n):
    root = DATA.parent
    for split in ("test", "valid"):
        files = sorted(p for p in (root / split / "images").glob("*")
                       if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
        if files:
            return np.stack([letterbox(f, IMGSZ) for f in files[:n]]), split
    raise SystemExit(f"❌ Aucune image dans {root}")


def copy_metadata(src, dst):
    """Recopie les métadonnées Ultralytics (noms des lettres, taille d'entrée…)."""
    m_src, m_dst = onnx.load(str(src)), onnx.load(str(dst))
    del m_dst.metadata_props[:]
    m_dst.metadata_props.extend(m_src.metadata_props)
    onnx.save(m_dst, str(dst))


def session(path):
    return ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])


def bench(sess, bs):
    name = sess.get_inputs()[0].name
    n = max(5, N_RUNS_BS1 // bs)
    x = np.random.rand(bs, 3, IMGSZ, IMGSZ).astype(np.float32)
    for _ in range(3):
        sess.run(None, {name: x})                     # warm-up
    t0 = time.perf_counter()
    for _ in range(n):
        sess.run(None, {name: x})
    return (time.perf_counter() - t0) / n * 1000 / bs  # ms par image


def evaluate(path, split):
    m = YOLO(str(path), task="detect").val(
        data=str(DATA), split=split, imgsz=IMGSZ, batch=1, device="cpu",
        workers=0, plots=False, verbose=False, project="runs/etape2", name="val", exist_ok=True)
    per_letter = {m.names[int(c)]: float(ap) for c, ap in zip(m.box.ap_class_index, m.box.ap50)}
    return float(m.box.map50), float(m.box.map), per_letter


# ---------------------------------------------------------------------------
def quantize(dst):
    """Essaie int8 signé (comme le TP). Certains CPU/ORT ne savent pas exécuter
    les convolutions int8 dynamiques (ConvInteger) : on retombe alors sur uint8."""
    for wtype in (QuantType.QInt8, QuantType.QUInt8):
        quantize_dynamic(str(FP32), str(dst), weight_type=wtype)
        try:
            session(dst)
            return wtype.name
        except Exception as e:
            print(f"⚠️  Poids {wtype.name} non exécutables par ONNX Runtime ({str(e)[:120]}…)")
    raise SystemExit("❌ Aucune version dynamique exécutable.")


def main():
    if not FP32.exists():
        raise SystemExit("❌ Lance d'abord l'étape 1 (exports\\asl_fp32_640.onnx manquant).")
    RESULTS.mkdir(exist_ok=True)

    # 1. Quantization dynamique
    dyn = OUT / "asl_int8_dyn_640.onnx"
    wtype = quantize(dyn)
    copy_metadata(FP32, dyn)
    print(f"\n✅ Modèle quantifié (poids {wtype}) : {dyn}")

    s32, s8 = session(FP32), session(dyn)
    name = s32.get_inputs()[0].name

    # 2. Écart des sorties sur de vraies images
    imgs, split = test_images(16)
    o32 = s32.run(None, {name: imgs})[0]
    o8 = s8.run(None, {name: imgs})[0]
    d = np.abs(o32[:, 4:, :] - o8[:, 4:, :])
    max_diff, mean_diff = float(d.max()), float(d.mean())

    # 3. Précision globale + par lettre
    print("\nÉvaluation de la précision…")
    m32 = evaluate(FP32, split)
    m8 = evaluate(dyn, split)

    # 4. Vitesse
    print("\nMesure de la vitesse sur CPU…")
    speed = {"fp32": {}, "int8_dyn": {}}
    for bs in BATCHES:
        speed["fp32"][bs] = bench(s32, bs)
        speed["int8_dyn"][bs] = bench(s8, bs)
        print(f"  batch {bs:>2} : FP32 {speed['fp32'][bs]:.1f} ms/img | "
              f"INT8 dyn {speed['int8_dyn'][bs]:.1f} ms/img")

    # 5. Lettres les plus touchées
    deltas = sorted(((l, m8[2].get(l, 0.0) - ap) for l, ap in m32[2].items()), key=lambda x: x[1])

    size32, size8 = FP32.stat().st_size / 1e6, dyn.stat().st_size / 1e6
    res = {
        "split": split, "poids_int8": wtype,
        "onnx_fp32": {"taille_mo": size32, "mAP50": m32[0], "mAP50-95": m32[1],
                      "ms_par_image": speed["fp32"], "ap50_par_lettre": m32[2]},
        "onnx_int8_dynamique": {"taille_mo": size8, "mAP50": m8[0], "mAP50-95": m8[1],
                                "ms_par_image": speed["int8_dyn"], "ap50_par_lettre": m8[2]},
        "ecart_sorties": {"max": max_diff, "moyen": mean_diff},
        "lettres_plus_degradees": deltas[:5],
    }
    (RESULTS / "etape2.json").write_text(json.dumps(res, indent=2))

    lines = [
        "", "=" * 66, "RÉSULTATS ÉTAPE 2 — ONNX FP32 vs INT8 dynamique (CPU)", "=" * 66,
        f"{'':20}{'FP32':>12}{'INT8 dyn':>12}{'Écart':>12}",
        f"{'Taille (Mo)':20}{size32:>12.2f}{size8:>12.2f}{f'÷{size32 / size8:.1f}':>12}",
        f"{'mAP50':20}{m32[0]:>12.3f}{m8[0]:>12.3f}{m8[0] - m32[0]:>+12.3f}",
        f"{'mAP50-95':20}{m32[1]:>12.3f}{m8[1]:>12.3f}{m8[1] - m32[1]:>+12.3f}",
    ]
    for bs in BATCHES:
        a, b = speed["fp32"][bs], speed["int8_dyn"][bs]
        lines.append(f"{f'ms/img batch={bs}':20}{a:>12.1f}{b:>12.1f}{f'×{a / b:.2f}':>12}")
    lines += [
        f"Écart des scores FP32/INT8 : max {max_diff:.2e} | moyen {mean_diff:.2e}",
        "Lettres les plus dégradées (AP50) : "
        + ", ".join(f"{l} {dv:+.3f}" for l, dv in deltas[:5]),
        f"Poids int8 utilisés : {wtype}   (split '{split}')", "=" * 66,
        f"Résultats sauvegardés dans {RESULTS / 'etape2.json'}",
    ]
    print("\n".join(lines))


if __name__ == "__main__":
    main()
