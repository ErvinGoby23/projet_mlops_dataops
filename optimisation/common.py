

import json
import statistics
import time
from pathlib import Path

import cv2
import numpy as np
import yaml

# Racine du projet (dossier parent du dossier contenant ce fichier)
ROOT = Path(__file__).resolve().parents[1]


# CONFIGURATION & GESTION DES FICHIERS
def load_config():
    """Charge config.yaml et convertit tous les chemins relatifs en chemins absolus."""
    cfg = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    cfg["paths"] = {k: ROOT / v for k, v in cfg["paths"].items()}
    return cfg


# Variable globale de configuration
CFG = load_config()
P = CFG["paths"]


def results_path(name):
    """Retourne le chemin d'un fichier dans results/ et cree le dossier si necessaire."""
    P["results"].mkdir(parents=True, exist_ok=True)
    return P["results"] / name


def save_json(name, obj):
    """Sauvegarde un objet sous forme de fichier JSON dans results/."""
    results_path(name).write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8")


def load_json(name):
    """Charge un fichier JSON depuis results/ (retourne None s'il n'existe pas)."""
    p = P["results"] / name
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def size_mb(path):
    """Calcule la taille d'un fichier en Mega-octets (Mo)."""
    return Path(path).stat().st_size / 1e6



# PRÉTRAITEMENT & PRÉPARATION DES DONNÉES
def letterbox(path, size):
    """
    Applique le meme pretraitement qu'Ultralytics : redimensionnement sans distorsion
    avec ajout de bandes grises pour obtenir une image carree de dimension (size, size).
    (Indispensable : un pretraitement different = train-serving skew.)
    """
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
    """Liste et trie les images (.jpg, .jpeg, .png) contenues dans un sous-dossier de dataset."""
    folder = P["data"].parent / split / "images"
    return sorted(p for p in folder.glob("*") if p.suffix.lower() in (".jpg", ".jpeg", ".png"))


def test_split():
    """Identifie le split a utiliser pour les tests (recherche 'test', bascule sur 'valid')."""
    s = CFG["evaluation"]["test_split"]
    return s if images_of(s) else "valid"


def batch_of(files, size):
    """Convertit et empile une liste d'images sous forme d'un unique ndarray NumPy (batch)."""
    return np.stack([letterbox(f, size) for f in files])


def eval_yaml():
    """
    Genere un fichier data_eval.yaml temporaire fusionnant valid + test (216 images).
    Jeu elargi plus robuste pour l'analyse par lettre.
    """
    cfg = yaml.safe_load(P["data"].read_text())
    root = P["data"].parent.resolve()
    tmp = root / "data_eval.yaml"
    tmp.write_text(yaml.safe_dump({
        "path": str(root),
        "train": "train/images",
        "val": "valid/images",
        "test": ["valid/images", "test/images"],
        "nc": cfg["nc"],
        "names": cfg["names"]
    }))
    return tmp



# ÉVALUATION DE LA PRÉCISION (mAP)
def evaluate(model_path, imgsz, data=None, split=None, tag="val", project="eval"):
    """
    Evalue la precision d'un modele via Ultralytics.
    Retourne un tuple : (mAP50, mAP50-95, dictionnaire AP50 par lettre).
    """
    from ultralytics import YOLO
    m = YOLO(str(model_path), task="detect").val(
        data=str(data or P["data"]),
        split=split or test_split(),
        imgsz=imgsz,
        batch=1,
        device=CFG["evaluation"]["device"],
        workers=0,
        plots=False,
        verbose=False,
        project=str(P["runs"] / project),
        name=tag,
        exist_ok=True
    )
    per_letter = {m.names[int(c)]: float(ap) for c, ap in zip(m.box.ap_class_index, m.box.ap50)}
    return float(m.box.map50), float(m.box.map), per_letter



# EXÉCUTION & BENCHMARK ONNX RUNTIME CPU
def session(path):
    """Instancie une session d'inference ONNX Runtime configuree pour une execution CPU."""
    import onnxruntime as ort
    return ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])


def input_name(path_or_sess):
    """Extrait le nom du premier tenseur d'entree de la session ONNX."""
    s = session(path_or_sess) if isinstance(path_or_sess, (str, Path)) else path_or_sess
    return s.get_inputs()[0].name


def bench(run_fn, imgsz, bs):
    """
    Chronomètre l'inference : phase d'echauffement (warm-up) puis calcul du temps moyen 
    par image en millisecondes.
    """
    b = CFG["benchmark"]
    n = max(5, b["runs_bs1"] // bs)
    x = np.random.rand(bs, 3, imgsz, imgsz).astype(np.float32)
    
    # Passages d'echauffement
    for _ in range(b["warmup"]):
        run_fn(x)
        
    # Mesure de performance
    t0 = time.perf_counter()
    for _ in range(n):
        run_fn(x)
    return (time.perf_counter() - t0) / n * 1000 / bs


def bench_onnx(path, imgsz, batches=None):
    """Mesure la latence d'un fichier ONNX pour differents batch sizes (par defaut 1 et 8)."""
    sess = session(path)
    name = sess.get_inputs()[0].name
    run = lambda x: sess.run(None, {name: x})
    out = {}
    for bs in batches or CFG["benchmark"]["batches"]:
        reps = CFG["benchmark"]["repeats_bs1"] if bs == 1 else 1
        out[bs] = statistics.median(bench(run, imgsz, bs) for _ in range(reps))
    return out


def score_gap(path_ref, path, imgs):
    """Calcule l'ecart maximal et moyen entre les sorties brutes d'un modele quantifie et sa reference FP32."""
    a = session(path_ref)
    b = session(path)
    name = a.get_inputs()[0].name
    d = np.abs(a.run(None, {name: imgs})[0][:, 4:, :] - b.run(None, {name: imgs})[0][:, 4:, :])
    return float(d.max()), float(d.mean())



# EXPORTATION & QUANTIFICATION STATIQUE QDQ
def export_onnx(imgsz, dst=None):
    """Exporte model/best.pt au format ONNX FP32 avec dimension dynamique."""
    from ultralytics import YOLO
    dst = Path(dst or P["exports"] / f"asl_fp32_{imgsz}.onnx")
    if not dst.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        p = YOLO(str(P["model_pt"])).export(
            format="onnx", imgsz=imgsz, dynamic=True, simplify=True, device="cpu"
        )
        Path(p).replace(dst)
    return dst


def copy_metadata(src, dst):
    """Recopie les metadonnees (noms des classes, ancres...) depuis la source FP32 vers le modele INT8."""
    import onnx
    m_src, m_dst = onnx.load(str(src)), onnx.load(str(dst))
    del m_dst.metadata_props[:]
    m_dst.metadata_props.extend(m_src.metadata_props)
    onnx.save(m_dst, str(dst))


def block_of(node_name):
    """Parse le nom d'un noeud ONNX pour extraire l'index de son bloc (ex: '/model.12/...' -> 12)."""
    if node_name.startswith("/model."):
        part = node_name.split("/")[1].split(".")[1]
        if part.isdigit():
            return int(part)
    return None


def head_nodes(onnx_path):
    """Liste les identifiants des noeuds non-convolutifs du bloc de tete (tete de detection)."""
    import onnx
    hb = CFG["quantization"]["head_block"]
    return [
        n.name for n in onnx.load(str(onnx_path)).graph.node
        if block_of(n.name) == hb and n.op_type != "Conv"
    ]


def calibration_files(n, seed=0):
    """Tire un echantillon aleatoire de N images dans le split d'entrainement pour la calibration."""
    import random
    files = images_of("train")
    random.Random(seed).shuffle(files)
    return files[:n]


class CalibReader:
    """Classe de base pour fournir les images de calibration une par une a ONNX Runtime."""
    def __init__(self, input_name, files, imgsz):
        self.input_name, self.files, self.imgsz, self.i = input_name, files, imgsz, 0

    def get_next(self):
        if self.i >= len(self.files):
            return None
        x = letterbox(self.files[self.i], self.imgsz)[None]
        self.i += 1
        return {self.input_name: x}


def quantize_static(src, dst, imgsz, calib_files, method="minmax", protect_head=True,
                    nodes_to_quantize=None, extra_options=None):
    """
    Applique une quantification statique post-entrainement QDQ (Quantize / Dequantize).
    Optionnellement, la tete de detection peut etre exclue de la quantification (protect_head=True).
    """
    from onnxruntime.quantization import (
        CalibrationDataReader,
        CalibrationMethod,
        QuantFormat,
        QuantType,
        quantize_static as _qs,
    )

    class _Reader(CalibReader, CalibrationDataReader):
        pass

    q = CFG["quantization"]
    methods = {
        "minmax": CalibrationMethod.MinMax,
        "percentile": CalibrationMethod.Percentile,
        "entropy": CalibrationMethod.Entropy,
    }
    
    _qs(
        str(src),
        str(dst),
        _Reader(input_name(src), calib_files, imgsz),
        quant_format=QuantFormat.QDQ,
        per_channel=q["per_channel"],
        activation_type=getattr(QuantType, q["activation_type"]),
        weight_type=getattr(QuantType, q["weight_type"]),
        calibrate_method=methods[method],
        nodes_to_quantize=nodes_to_quantize,
        nodes_to_exclude=head_nodes(src) if protect_head else None,
        extra_options=extra_options or {},
    )
    copy_metadata(src, dst)
    return Path(dst)



# AFFICHAGE

def banner(title, width=78):
    """Construit un en-tete formate de lignes pour l'affichage console."""
    return ["", "=" * width, title, "=" * width]