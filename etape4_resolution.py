"""
ÉTAPE 4 — 2e optimisation : réduire la résolution d'entrée (640 -> 416 -> 320)

Idée : en 320x320 il y a 4x moins de pixels qu'en 640x640, donc ~4x moins de calcul.
Le modèle a été entraîné en 640 : on mesure combien de précision on perd
SANS ré-entraîner. On combine avec la meilleure quantization de l'étape 3
(statique MinMax, tête FP32 = variante B).

Versions comparées (toutes mesurées dans le même run, pour être équitable) :
  640 FP32 | 640 INT8 | 416 FP32 | 416 INT8 | 320 FP32 | 320 INT8

Lancement :  python .\\etape4_resolution.py      (~5-10 min)
"""
import json
import random
import statistics
import time
from pathlib import Path

import cv2
import numpy as np
import onnx
import onnxruntime as ort
import yaml
from onnxruntime.quantization import (CalibrationDataReader, CalibrationMethod, QuantFormat,
                                      QuantType, quantize_static)
from ultralytics import YOLO

MODEL_PT = Path(r"model\best.pt")
DATA = Path(r"datasets\asl\data.yaml")
OUT = Path("exports")
RESULTS = Path("results")
SIZES = [640, 416, 320]
N_CALIB = 100
BATCHES = [1, 8]
N_RUNS_BS1 = 50
REPEATS_BS1 = 3


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


def images_of(split):
    return sorted(p for p in (DATA.parent / split / "images").glob("*")
                  if p.suffix.lower() in (".jpg", ".jpeg", ".png"))


class CalibReader(CalibrationDataReader):
    def __init__(self, input_name, files, size):
        self.input_name, self.files, self.size, self.i = input_name, files, size, 0

    def get_next(self):
        if self.i >= len(self.files):
            return None
        x = letterbox(self.files[self.i], self.size)[None]
        self.i += 1
        return {self.input_name: x}


def copy_metadata(src, dst):
    m_src, m_dst = onnx.load(str(src)), onnx.load(str(dst))
    del m_dst.metadata_props[:]
    m_dst.metadata_props.extend(m_src.metadata_props)
    onnx.save(m_dst, str(dst))


def export_fp32(size):
    dst = OUT / f"asl_fp32_{size}.onnx"
    if not dst.exists():
        p = YOLO(str(MODEL_PT)).export(format="onnx", imgsz=size, dynamic=True,
                                       simplify=True, device="cpu")
        Path(p).replace(dst)
    return dst


def quantize_b(fp32, size, calib_files):
    """Même recette que la variante B de l'étape 3 (MinMax, per-channel, tête FP32)."""
    dst = OUT / f"asl_int8_static_{size}.onnx"
    graph = onnx.load(str(fp32)).graph
    head = [n.name for n in graph.node if "/model.22/" in n.name and n.op_type != "Conv"]
    quantize_static(str(fp32), str(dst), CalibReader(graph.input[0].name, calib_files, size),
                    quant_format=QuantFormat.QDQ, per_channel=True,
                    activation_type=QuantType.QUInt8, weight_type=QuantType.QInt8,
                    calibrate_method=CalibrationMethod.MinMax, nodes_to_exclude=head)
    copy_metadata(fp32, dst)
    return dst


def eval_yaml():
    cfg = yaml.safe_load(DATA.read_text())
    root = DATA.parent.resolve()
    tmp = root / "data_eval.yaml"
    tmp.write_text(yaml.safe_dump({
        "path": str(root), "train": "train/images", "val": "valid/images",
        "test": ["valid/images", "test/images"], "nc": cfg["nc"], "names": cfg["names"]}))
    return tmp


def evaluate(path, data, size, tag):
    m = YOLO(str(path), task="detect").val(
        data=str(data), split="test", imgsz=size, batch=1, device="cpu", workers=0,
        plots=False, verbose=False, project="runs/etape4", name=tag, exist_ok=True)
    per_letter = {m.names[int(c)]: float(ap) for c, ap in zip(m.box.ap_class_index, m.box.ap50)}
    return float(m.box.map50), float(m.box.map), per_letter


def bench(path, size, bs):
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    name = sess.get_inputs()[0].name
    n = max(5, N_RUNS_BS1 // bs)
    x = np.random.rand(bs, 3, size, size).astype(np.float32)
    for _ in range(3):
        sess.run(None, {name: x})
    t0 = time.perf_counter()
    for _ in range(n):
        sess.run(None, {name: x})
    return (time.perf_counter() - t0) / n * 1000 / bs


# ---------------------------------------------------------------------------
def main():
    OUT.mkdir(exist_ok=True)
    RESULTS.mkdir(exist_ok=True)
    random.seed(0)
    calib = random.sample(images_of("train"), min(N_CALIB, len(images_of("train"))))
    data_eval = eval_yaml()

    models = {}
    for size in SIZES:
        print(f"\n=== Résolution {size} ===")
        fp32 = export_fp32(size)
        int8 = quantize_b(fp32, size, calib)
        models[f"{size}_FP32"] = (fp32, size)
        models[f"{size}_INT8"] = (int8, size)
        print(f"✅ {fp32.name} ({fp32.stat().st_size / 1e6:.2f} Mo) | "
              f"{int8.name} ({int8.stat().st_size / 1e6:.2f} Mo)")

    res = {}
    for name, (path, size) in models.items():
        print(f"\nÉvaluation {name}…")
        t = evaluate(path, DATA, size, f"{name}_test")
        e = evaluate(path, data_eval, size, f"{name}_eval")
        res[name] = {"resolution": size, "taille_mo": path.stat().st_size / 1e6,
                     "mAP50_test": t[0], "mAP50-95_test": t[1],
                     "mAP50_216img": e[0], "mAP50-95_216img": e[1], "ap50_par_lettre": e[2]}

    print("\nMesure de la vitesse sur CPU…")
    for name, (path, size) in models.items():
        res[name]["ms_par_image"] = {}
        for bs in BATCHES:
            reps = REPEATS_BS1 if bs == 1 else 1
            res[name]["ms_par_image"][bs] = statistics.median(bench(path, size, bs) for _ in range(reps))
        print(f"  {name:10} b1: {res[name]['ms_par_image'][1]:.1f} ms | b8: {res[name]['ms_par_image'][8]:.1f} ms")

    # Lettres les plus dégradées : 320 INT8 vs 640 FP32 (la combinaison la plus agressive)
    ref = res["640_FP32"]["ap50_par_lettre"]
    agg = res["320_INT8"]["ap50_par_lettre"]
    deltas = sorted(((l, agg.get(l, 0.0) - ap) for l, ap in ref.items()), key=lambda x: x[1])
    res["_lettres_plus_degradees_320_INT8"] = deltas[:5]
    (RESULTS / "etape4.json").write_text(json.dumps(res, indent=2, default=str))

    base = res["640_FP32"]
    lines = ["", "=" * 86, "RÉSULTATS ÉTAPE 4 — Résolution × quantization (CPU)", "=" * 86,
             f"{'':12}{'Taille':>8}{'mAP50':>8}{'mAP50-95':>10}{'mAP50':>8}{'ms b1':>8}{'ms b8':>8}"
             f"{'Accél.':>8}{'Perte':>9}",
             f"{'':12}{'(Mo)':>8}{'test':>8}{'test':>10}{'216img':>8}{'':>8}{'':>8}{'b1':>8}{'mAP50':>9}"]
    for name in models:
        r = res[name]
        accel = "x{:.2f}".format(base["ms_par_image"][1] / r["ms_par_image"][1])
        lines.append(
            f"{name:12}{r['taille_mo']:>8.2f}{r['mAP50_test']:>8.3f}{r['mAP50-95_test']:>10.3f}"
            f"{r['mAP50_216img']:>8.3f}{r['ms_par_image'][1]:>8.1f}{r['ms_par_image'][8]:>8.1f}"
            f"{accel:>8}{r['mAP50_216img'] - base['mAP50_216img']:>+9.3f}")
    lines += ["",
              "Lettres les plus dégradées (320 INT8 vs 640 FP32, AP50 216 img) : "
              + ", ".join(f"{l} {d:+.3f}" for l, d in deltas[:5]),
              "(Perte mAP50 calculée sur le jeu élargi de 216 images)",
              "=" * 86, f"Résultats sauvegardés dans {RESULTS / 'etape4.json'}"]
    print("\n".join(lines))


if __name__ == "__main__":
    main()
