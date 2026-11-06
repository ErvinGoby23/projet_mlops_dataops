import json
import statistics
import time
from pathlib import Path

import cv2
import numpy as np
import yaml

# Dossier racine du projet
ROOT = Path(__file__).resolve().parents[1]


# Configuration

# Lit config.yaml et transforme les chemins en chemins complets
def load_config():
    cfg = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    cfg["paths"] = {k: ROOT / v for k, v in cfg["paths"].items()}
    return cfg


# Configuration chargée une seule fois
CFG = load_config()
P = CFG["paths"]


# Donne le chemin d'un fichier dans le dossier des résultats (crée le dossier si besoin)
def results_path(name):
    P["results"].mkdir(parents=True, exist_ok=True)
    return P["results"] / name


# Enregistre un objet dans un fichier JSON
def save_json(name, obj):
    results_path(name).write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8")


# Relit un fichier JSON (renvoie None s'il n'existe pas)
def load_json(name):
    p = P["results"] / name
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


# Taille d'un fichier en Mo
def size_mb(path):
    return Path(path).stat().st_size / 1e6

# Prépare une image comme Ultralytics : redimensionne sans déformer et ajoute des bandes grises
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


# Liste les images d'un dossier (train, valid ou test)
def images_of(split):
    folder = P["data"].parent / split / "images"
    return sorted(p for p in folder.glob("*") if p.suffix.lower() in (".jpg", ".jpeg", ".png"))


# Choisit le dossier de test, ou valid si test est vide
def test_split():
    s = CFG["evaluation"]["test_split"]
    return s if images_of(s) else "valid"


# Prépare plusieurs images et les regroupe en un seul lot
def batch_of(files, size):
    return np.stack([letterbox(f, size) for f in files])


# Crée un fichier de données qui regroupe valid et test pour l'évaluation
def eval_yaml():
    cfg = yaml.safe_load(P["data"].read_text())
    root = P["data"].parent.resolve()
    tmp = root / "data_eval.yaml"
    tmp.write_text(yaml.safe_dump({
        "path": str(root), "train": "train/images", "val": "valid/images",
        "test": ["valid/images", "test/images"], "nc": cfg["nc"], "names": cfg["names"]}))
    return tmp


# Évaluation (précision)

# Mesure la précision du modèle : mAP50, mAP50-95 et score par lettre
def evaluate(model_path, imgsz, data=None, split=None, tag="val", project="eval"):
    from ultralytics import YOLO
    m = YOLO(str(model_path), task="detect").val(
        data=str(data or P["data"]), split=split or test_split(), imgsz=imgsz, batch=1,
        device=CFG["evaluation"]["device"], workers=0, plots=False, verbose=False,
        project=str(P["runs"] / project), name=tag, exist_ok=True)
    per_letter = {m.names[int(c)]: float(ap) for c, ap in zip(m.box.ap_class_index, m.box.ap50)}
    return float(m.box.map50), float(m.box.map), per_letter

# Ouvre le modèle ONNX sur le processeur
def session(path):
    import onnxruntime as ort
    return ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])


# Donne le nom de l'entrée du modèle
def input_name(path_or_sess):
    s = session(path_or_sess) if isinstance(path_or_sess, (str, Path)) else path_or_sess
    return s.get_inputs()[0].name


# Mesure le temps moyen par image en millisecondes (avec un échauffement avant)
def bench(run_fn, imgsz, bs):
    b = CFG["benchmark"]
    n = max(5, b["runs_bs1"] // bs)
    x = np.random.rand(bs, 3, imgsz, imgsz).astype(np.float32)
    for _ in range(b["warmup"]):
        run_fn(x)
    t0 = time.perf_counter()
    for _ in range(n):
        run_fn(x)
    return (time.perf_counter() - t0) / n * 1000 / bs


# Mesure la vitesse du modèle ONNX pour chaque taille de lot
def bench_onnx(path, imgsz, batches=None):
    sess = session(path)
    name = sess.get_inputs()[0].name
    run = lambda x: sess.run(None, {name: x})  # noqa: E731
    out = {}
    for bs in batches or CFG["benchmark"]["batches"]:
        reps = CFG["benchmark"]["repeats_bs1"] if bs == 1 else 1
        out[bs] = statistics.median(bench(run, imgsz, bs) for _ in range(reps))
    return out


# Compare les scores des lettres entre deux modèles (écart max et écart moyen)
def score_gap(path_ref, path, imgs):
    a = session(path_ref)
    b = session(path)
    name = a.get_inputs()[0].name
    d = np.abs(a.run(None, {name: imgs})[0][:, 4:, :] - b.run(None, {name: imgs})[0][:, 4:, :])
    return float(d.max()), float(d.mean())



# Convertit le modèle best.pt en ONNX (ne refait rien si le fichier existe déjà)
def export_onnx(imgsz, dst=None):
    from ultralytics import YOLO
    dst = Path(dst or P["exports"] / f"asl_fp32_{imgsz}.onnx")
    if not dst.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        p = YOLO(str(P["model_pt"])).export(format="onnx", imgsz=imgsz, dynamic=True,
                                            simplify=True, device="cpu")
        Path(p).replace(dst)
    return dst


# Recopie les infos du modèle (noms des lettres, taille) perdues pendant la quantization
def copy_metadata(src, dst):
    import onnx
    m_src, m_dst = onnx.load(str(src)), onnx.load(str(dst))
    del m_dst.metadata_props[:]
    m_dst.metadata_props.extend(m_src.metadata_props)
    onnx.save(m_dst, str(dst))


# Trouve le numéro du bloc à partir du nom d'un nœud
def block_of(node_name):
    if node_name.startswith("/model."):
        part = node_name.split("/")[1].split(".")[1]
        if part.isdigit():
            return int(part)
    return None


# Liste les nœuds de la tête du modèle à garder en FP32
def head_nodes(onnx_path):
    import onnx
    hb = CFG["quantization"]["head_block"]
    return [n.name for n in onnx.load(str(onnx_path)).graph.node
            if block_of(n.name) == hb and n.op_type != "Conv"]


# Choisit au hasard des images d'entraînement pour la calibration
def calibration_files(n, seed=0):
    import random
    files = images_of("train")              # jamais les images de test
    random.Random(seed).shuffle(files)
    return files[:n]


# Donne les images de calibration une par une
class CalibReader:
    # Garde en mémoire les images et la position actuelle
    def __init__(self, input_name, files, imgsz):
        self.input_name, self.files, self.imgsz, self.i = input_name, files, imgsz, 0

    # Renvoie l'image suivante, ou None quand il n'y en a plus
    def get_next(self):
        if self.i >= len(self.files):
            return None
        x = letterbox(self.files[self.i], self.imgsz)[None]
        self.i += 1
        return {self.input_name: x}


# Réduit le modèle en INT8 pour le rendre plus léger et plus rapide
def quantize_static(src, dst, imgsz, calib_files, method="minmax", protect_head=True,
                    nodes_to_quantize=None, extra_options=None):
    from onnxruntime.quantization import (CalibrationDataReader, CalibrationMethod, QuantFormat,
                                          QuantType, quantize_static as _qs)

    # Lecteur d'images compatible avec ONNX Runtime
    class _Reader(CalibReader, CalibrationDataReader):
        pass

    q = CFG["quantization"]
    methods = {"minmax": CalibrationMethod.MinMax, "percentile": CalibrationMethod.Percentile,
               "entropy": CalibrationMethod.Entropy}
    _qs(str(src), str(dst), _Reader(input_name(src), calib_files, imgsz),
        quant_format=QuantFormat.QDQ, per_channel=q["per_channel"],
        activation_type=getattr(QuantType, q["activation_type"]),
        weight_type=getattr(QuantType, q["weight_type"]),
        nodes_to_quantize=nodes_to_quantize,
        calibrate_method=methods[method],
        nodes_to_exclude=head_nodes(src) if protect_head else None,
        extra_options=extra_options or {})
    copy_metadata(src, dst)
    return Path(dst)

# Affichage

# Crée un titre encadré par des lignes de =
def banner(title, width=78):
    return ["", "=" * width, title, "=" * width]