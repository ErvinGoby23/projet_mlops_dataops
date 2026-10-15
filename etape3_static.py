"""
ÉTAPE 3 — Quantization post-training STATIQUE (calibrée)

Différence avec l'étape 2 : les échelles (s) et zero-points (z) des ACTIVATIONS
sont calculés UNE FOIS à l'avance, en faisant passer des images d'entraînement
dans le modèle (calibration). À l'inférence, plus rien n'est calculé à la volée
-> ONNX Runtime peut utiliser des convolutions int8 rapides (QLinearConv).

3 variantes comparées à ONNX FP32 :
  A. MinMax,     tout quantifié          -> version "naïve"
  B. MinMax,     tête de détection FP32  -> protéger la dernière couche aide-t-il ?
  C. Percentile, tête de détection FP32  -> ignorer les valeurs extrêmes (outliers) aide-t-il ?

Lancement :  python .\\etape3_static.py      (~10-15 min)
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
from onnxruntime.quantization.shape_inference import quant_pre_process
from ultralytics import YOLO

DATA = Path(r"datasets\asl\data.yaml")
FP32 = Path(r"exports\asl_fp32_640.onnx")
OUT = Path("exports")
RESULTS = Path("results")
IMGSZ = 640
N_CALIB = 100            # images d'ENTRAÎNEMENT pour la calibration (jamais celles de test)
BATCHES = [1, 8, 32]
N_RUNS_BS1 = 50
REPEATS_BS1 = 3          # on répète la mesure batch=1 et on garde la médiane

VARIANTS = {
    "A_minmax_tout":       dict(method=CalibrationMethod.MinMax, protect_head=False, extra={}),
    "B_minmax_tete_fp32":  dict(method=CalibrationMethod.MinMax, protect_head=True, extra={}),
    "C_percentile_tete_fp32": dict(method=CalibrationMethod.Percentile, protect_head=True,
                                   extra={"CalibPercentile": 99.999}),
}


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
    """Fournit les images de calibration une par une à ONNX Runtime."""
    def __init__(self, input_name, files):
        self.input_name, self.files, self.i = input_name, files, 0

    def get_next(self):
        if self.i >= len(self.files):
            return None
        x = letterbox(self.files[self.i], IMGSZ)[None]
        self.i += 1
        return {self.input_name: x}


def copy_metadata(src, dst):
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
        sess.run(None, {name: x})
    t0 = time.perf_counter()
    for _ in range(n):
        sess.run(None, {name: x})
    return (time.perf_counter() - t0) / n * 1000 / bs


def eval_yaml():
    """Jeu élargi valid+test (216 images) pour l'analyse par lettre : plus robuste que
    les 2-3 images par lettre du seul jeu de test."""
    cfg = yaml.safe_load(DATA.read_text())
    root = DATA.parent.resolve()
    tmp = root / "data_eval.yaml"
    tmp.write_text(yaml.safe_dump({
        "path": str(root), "train": "train/images", "val": "valid/images",
        "test": ["valid/images", "test/images"], "nc": cfg["nc"], "names": cfg["names"]}))
    return tmp


def evaluate(path, data, split, tag):
    m = YOLO(str(path), task="detect").val(
        data=str(data), split=split, imgsz=IMGSZ, batch=1, device="cpu", workers=0,
        plots=False, verbose=False, project="runs/etape3", name=tag, exist_ok=True)
    per_letter = {m.names[int(c)]: float(ap) for c, ap in zip(m.box.ap_class_index, m.box.ap50)}
    return float(m.box.map50), float(m.box.map), per_letter


# ---------------------------------------------------------------------------
def main():
    if not FP32.exists():
        raise SystemExit("❌ Lance d'abord l'étape 1.")
    RESULTS.mkdir(exist_ok=True)

    # 0. Pré-traitement recommandé par ONNX Runtime (fusion d'opérations, formes)
    pre = OUT / "asl_fp32_640_pre.onnx"
    try:
        quant_pre_process(str(FP32), str(pre))
        print("✅ Pré-traitement ONNX effectué")
    except Exception as e:
        print(f"⚠️  Pré-traitement impossible ({e}) -> on quantifie le modèle brut")
        pre = FP32

    graph = onnx.load(str(pre)).graph
    input_name = graph.input[0].name
    head_nodes = [n.name for n in graph.node if "/model.22/" in n.name and n.op_type != "Conv"]
    print(f"Tête de détection : {len(head_nodes)} nœuds non-convolution gardables en FP32")

    random.seed(0)
    calib_files = random.sample(images_of("train"), min(N_CALIB, len(images_of("train"))))
    print(f"Calibration sur {len(calib_files)} images d'entraînement")

    # 1. Quantization des 3 variantes
    paths = {}
    for name, cfg in VARIANTS.items():
        dst = OUT / f"asl_int8_static_{name}.onnx"
        print(f"\n=== Quantization {name} ===")
        t0 = time.perf_counter()
        quantize_static(
            str(pre), str(dst), CalibReader(input_name, calib_files),
            quant_format=QuantFormat.QDQ, per_channel=True,
            activation_type=QuantType.QUInt8, weight_type=QuantType.QInt8,
            calibrate_method=cfg["method"],
            nodes_to_exclude=head_nodes if cfg["protect_head"] else None,
            extra_options=cfg["extra"])
        copy_metadata(FP32, dst)
        paths[name] = dst
        print(f"✅ {dst.name} ({dst.stat().st_size / 1e6:.2f} Mo, {time.perf_counter() - t0:.0f} s)")

    # 2. Évaluation : test (comparable aux étapes 1-2) + jeu élargi (analyse par lettre)
    data_eval = eval_yaml()
    test_split = "test" if images_of("test") else "val"
    all_models = {"FP32": FP32, **paths}
    res = {}
    for name, p in all_models.items():
        print(f"\nÉvaluation {name}…")
        t = evaluate(p, DATA, test_split, f"{name}_test")
        e = evaluate(p, data_eval, "test", f"{name}_eval")
        res[name] = {"taille_mo": p.stat().st_size / 1e6,
                     "mAP50_test": t[0], "mAP50-95_test": t[1],
                     "mAP50_216img": e[0], "mAP50-95_216img": e[1], "ap50_par_lettre": e[2]}

    # 3. Vitesse (batch=1 répété, médiane)
    print("\nMesure de la vitesse sur CPU…")
    sessions = {n: session(p) for n, p in all_models.items()}
    for name, s in sessions.items():
        res[name]["ms_par_image"] = {}
        for bs in BATCHES:
            reps = REPEATS_BS1 if bs == 1 else 1
            vals = [bench(s, bs) for _ in range(reps)]
            res[name]["ms_par_image"][bs] = statistics.median(vals)
        print(f"  {name:24} " + " | ".join(f"b{bs}: {res[name]['ms_par_image'][bs]:.1f} ms"
                                           for bs in BATCHES))

    # 4. Écart des sorties vs FP32 (16 images de test)
    imgs = np.stack([letterbox(f, IMGSZ) for f in images_of(test_split)[:16]])
    ref = sessions["FP32"].run(None, {input_name: imgs})[0]
    for name in paths:
        out = sessions[name].run(None, {input_name: imgs})[0]
        d = np.abs(ref[:, 4:, :] - out[:, 4:, :])
        res[name]["ecart_max"], res[name]["ecart_moyen"] = float(d.max()), float(d.mean())

    # 5. Lettres les plus dégradées pour la meilleure variante (mAP50 jeu élargi)
    best = max(paths, key=lambda n: res[n]["mAP50_216img"])
    ref_l = res["FP32"]["ap50_par_lettre"]
    deltas = sorted(((l, res[best]["ap50_par_lettre"].get(l, 0.0) - ap) for l, ap in ref_l.items()),
                    key=lambda x: x[1])
    res["_meilleure_variante"] = best
    res["_lettres_plus_degradees"] = deltas[:5]
    (RESULTS / "etape3.json").write_text(json.dumps(res, indent=2, default=str))

    # 6. Tableau
    f = res["FP32"]
    hdr = f"{'':26}{'Taille':>8}{'mAP50':>8}{'mAP50':>8}{'ms b1':>8}{'ms b8':>8}{'ms b32':>8}{'Écart':>10}"
    sub = f"{'':26}{'(Mo)':>8}{'test':>8}{'216img':>8}{'':>8}{'':>8}{'':>8}{'max':>10}"
    lines = ["", "=" * 84, "RÉSULTATS ÉTAPE 3 — Quantization statique (CPU)", "=" * 84, hdr, sub]
    for name in all_models:
        r = res[name]
        ms = r["ms_par_image"]
        ecart = f"{r['ecart_max']:.2e}" if name != "FP32" else "—"
        lines.append(f"{name:26}{r['taille_mo']:>8.2f}{r['mAP50_test']:>8.3f}{r['mAP50_216img']:>8.3f}"
                     f"{ms[1]:>8.1f}{ms[8]:>8.1f}{ms[32]:>8.1f}{ecart:>10}")
    lines += [
        "",
        f"Accélération batch 1 vs FP32 : " + ", ".join(
            f"{n.split('_')[0]} ×{f['ms_par_image'][1] / res[n]['ms_par_image'][1]:.2f}" for n in paths),
        f"Meilleure variante (précision) : {best}",
        "Lettres les plus dégradées (AP50, 216 img) : "
        + ", ".join(f"{l} {d:+.3f}" for l, d in deltas[:5]),
        "=" * 84, f"Résultats sauvegardés dans {RESULTS / 'etape3.json'}",
    ]
    print("\n".join(lines))


if __name__ == "__main__":
    main()
