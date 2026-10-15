"""
ÉTAPE 5 — MLflow : tracer toutes les versions + graphiques + règle de décision

Rappel du cours : "ML system source code = training code + data".
On versionne donc ensemble : le DATASET (empreinte), les MODÈLES et les MÉTRIQUES.

  1. Lit les résultats des étapes 1 à 4 (results/etape*.json)
  2. Crée un run MLflow par version (paramètres, métriques, fichier du modèle)
  3. Applique la règle de décision fixée à l'avance
  4. Enregistre le modèle choisi dans le Model Registry (alias "production")
  5. Génère les graphiques pour le rapport / les slides (results/graphiques/)

Lancement :
    pip install mlflow matplotlib
    python .\\etape5_mlflow.py
Puis pour voir l'interface :
    mlflow ui --backend-store-uri sqlite:///mlflow.db
    -> ouvrir http://127.0.0.1:5000
"""
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

RESULTS = Path("results")
CHARTS = RESULTS / "graphiques"
EXPORTS = Path("exports")
DATA_DIR = Path(r"datasets\asl")
MAX_DROP = 0.02          # règle : perte max de 2 points de mAP50 vs la référence 640 FP32
REFERENCE = "onnx_fp32_640"
TRACKING_URI = "sqlite:///mlflow.db"
EXPERIMENT = "asl-embarque-quantization"
REGISTERED_NAME = "asl-detector"


# ---------------------------------------------------------------------------
def load(n):
    p = RESULTS / f"etape{n}.json"
    return json.loads(p.read_text()) if p.exists() else None


def ms(d, bs):
    """Les clés JSON sont des chaînes ("1", "8"…)."""
    if not d:
        return None
    return d.get(str(bs), d.get(bs))


def collect():
    """Construit une liste unique de versions à partir des 4 étapes."""
    v = {}
    e1, e2, e3, e4 = load(1), load(2), load(3), load(4)

    if e1:
        p = e1["pytorch"]
        v["pytorch_fp32_640"] = dict(etape=1, format="PyTorch", precision="FP32*", methode="aucune",
                                     imgsz=640, taille=p["taille_mo"], map50_test=p["mAP50"],
                                     map5095_test=p["mAP50-95"], ms1=ms(p["ms_par_image"], 1),
                                     ms8=ms(p["ms_par_image"], 8), fichier=Path(r"model\best.pt"))
    if e2:
        d = e2["onnx_int8_dynamique"]
        v["onnx_int8_dyn_640"] = dict(etape=2, format="ONNX", precision="INT8", methode="PTQ dynamique",
                                      imgsz=640, taille=d["taille_mo"], map50_test=d["mAP50"],
                                      map5095_test=d["mAP50-95"], ms1=ms(d["ms_par_image"], 1),
                                      ms8=ms(d["ms_par_image"], 8),
                                      fichier=EXPORTS / "asl_int8_dyn_640.onnx")
    if e3:
        for key, label, meth in [("A_minmax_tout", "onnx_int8_static_A_640", "PTQ statique MinMax, tout quantifié"),
                                 ("C_percentile_tete_fp32", "onnx_int8_static_C_640", "PTQ statique Percentile, tête FP32")]:
            if key in e3:
                d = e3[key]
                v[label] = dict(etape=3, format="ONNX", precision="INT8", methode=meth, imgsz=640,
                                taille=d["taille_mo"], map50_test=d["mAP50_test"],
                                map5095_test=d["mAP50-95_test"], map50_216=d["mAP50_216img"],
                                ms1=ms(d["ms_par_image"], 1), ms8=ms(d["ms_par_image"], 8),
                                lettres=d.get("ap50_par_lettre"),
                                fichier=EXPORTS / f"asl_int8_static_{key}.onnx")
    if e4:
        for size in (640, 416, 320):
            for prec in ("FP32", "INT8"):
                d = e4.get(f"{size}_{prec}")
                if not d:
                    continue
                name = f"onnx_{prec.lower()}_{size}" if prec == "FP32" else f"onnx_int8_static_{size}"
                v[name] = dict(etape=4, format="ONNX", precision=prec,
                               methode="aucune" if prec == "FP32" else "PTQ statique MinMax, tête FP32",
                               imgsz=size, taille=d["taille_mo"], map50_test=d["mAP50_test"],
                               map5095_test=d["mAP50-95_test"], map50_216=d["mAP50_216img"],
                               ms1=ms(d["ms_par_image"], 1), ms8=ms(d["ms_par_image"], 8),
                               lettres=d.get("ap50_par_lettre"),
                               fichier=EXPORTS / (f"asl_fp32_{size}.onnx" if prec == "FP32"
                                                  else f"asl_int8_static_{size}.onnx"))
    return v


def dataset_fingerprint():
    """Empreinte du dataset : si une image change, l'empreinte change."""
    h = hashlib.sha256()
    counts = {}
    for split in ("train", "valid", "test"):
        files = sorted((DATA_DIR / split / "images").glob("*"))
        counts[split] = len(files)
        for f in files:
            h.update(f.name.encode())
            h.update(str(f.stat().st_size).encode())
    return h.hexdigest()[:16], counts


# ---------------------------------------------------------------------------
def decide(v):
    ref = v[REFERENCE]
    key = "map50_216" if "map50_216" in ref else "map50_test"
    cands = []
    for name, r in v.items():
        if r["precision"] != "INT8" or r.get(key) is None or r.get("ms1") is None:
            continue
        r["perte"] = r[key] - ref[key]
        r["ok"] = r["perte"] >= -MAX_DROP
        if r.get("lettres") and ref.get("lettres"):
            worst = min(((l, r["lettres"].get(l, 0) - ap) for l, ap in ref["lettres"].items()),
                        key=lambda x: x[1])
            r["pire_lettre"] = worst
        cands.append(name)
    ok = [n for n in cands if v[n]["ok"]]
    choice = min(ok, key=lambda n: v[n]["ms1"]) if ok else None
    return choice, cands, key


def log_mlflow(v, choice, fp, counts):
    try:
        import mlflow
        from mlflow.tracking import MlflowClient
    except ImportError:
        print("⚠️  mlflow non installé : pip install mlflow")
        return
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT)
    for name, r in v.items():
        with mlflow.start_run(run_name=name) as run:
            mlflow.set_tags({"etape": r["etape"], "dataset_empreinte": fp,
                             "choisi": str(name == choice)})
            mlflow.log_params({"format": r["format"], "precision": r["precision"],
                               "methode": r["methode"], "imgsz": r["imgsz"],
                               "dataset_train": counts["train"], "dataset_valid": counts["valid"],
                               "dataset_test": counts["test"]})
            metrics = {"taille_mo": r["taille"], "mAP50_test": r["map50_test"],
                       "mAP50-95_test": r["map5095_test"], "ms_image_batch1": r["ms1"],
                       "ms_image_batch8": r["ms8"], "mAP50_216img": r.get("map50_216"),
                       "perte_mAP50": r.get("perte")}
            mlflow.log_metrics({k: float(x) for k, x in metrics.items() if x is not None})

            f = Path(r["fichier"])
            if name == choice and f.suffix == ".onnx" and f.exists():
                try:
                    import onnx
                    import mlflow.onnx
                    try:
                        info = mlflow.onnx.log_model(onnx.load(str(f)), name="model",
                                                     registered_model_name=REGISTERED_NAME)
                    except TypeError:
                        info = mlflow.onnx.log_model(onnx.load(str(f)), artifact_path="model",
                                                     registered_model_name=REGISTERED_NAME)
                    client = MlflowClient()
                    version = getattr(info, "registered_model_version", None) or max(
                        int(m.version) for m in client.search_model_versions(f"name='{REGISTERED_NAME}'"))
                    client.set_registered_model_alias(REGISTERED_NAME, "production", str(version))
                    print(f"✅ {name} enregistré : models:/{REGISTERED_NAME}@production (v{version})")
                except Exception as e:
                    print(f"⚠️  Enregistrement dans le registry impossible : {e}")
                    mlflow.log_artifact(str(f), artifact_path="model")
            elif f.exists():
                mlflow.log_artifact(str(f), artifact_path="model")
        print(f"  run MLflow : {name}")


def charts(v, choice, key):
    CHARTS.mkdir(parents=True, exist_ok=True)
    pts = {n: r for n, r in v.items() if r.get("ms1") and r.get(key, r["map50_test"]) is not None}

    # 1. Vitesse vs précision
    fig, ax = plt.subplots(figsize=(8, 5))
    for n, r in pts.items():
        y = r.get(key) if r.get(key) is not None else r["map50_test"]
        color = "#d9480f" if n == choice else ("#1c7ed6" if r["precision"] == "INT8" else "#868e96")
        ax.scatter(r["ms1"], y, s=r["taille"] * 25, color=color, alpha=0.8)
        ax.annotate(n.replace("onnx_", ""), (r["ms1"], y), fontsize=8, xytext=(5, 4),
                    textcoords="offset points")
    ref = v[REFERENCE]
    ax.axhline(ref[key] - MAX_DROP, color="#e03131", ls="--", lw=1, label=f"seuil (−{MAX_DROP:.2f})")
    ax.set_xlabel("ms par image, batch 1 (CPU) — plus à gauche = plus rapide")
    ax.set_ylabel("mAP50" + (" (216 images)" if key == "map50_216" else " (test)"))
    ax.set_title("Compromis vitesse / précision (taille du point = taille du modèle)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(CHARTS / "vitesse_vs_precision.png", dpi=150)

    # 2. Taille
    fig, ax = plt.subplots(figsize=(8, 4))
    names = list(v)
    ax.barh([n.replace("onnx_", "") for n in names], [v[n]["taille"] for n in names],
            color=["#d9480f" if n == choice else "#1c7ed6" for n in names])
    ax.set_xlabel("Taille du fichier (Mo)")
    ax.set_title("Taille des modèles")
    fig.tight_layout()
    fig.savefig(CHARTS / "taille.png", dpi=150)

    # 3. Écart par lettre du modèle choisi
    if choice and v[choice].get("lettres") and ref.get("lettres"):
        letters = sorted(ref["lettres"])
        d = [v[choice]["lettres"].get(l, 0) - ref["lettres"][l] for l in letters]
        fig, ax = plt.subplots(figsize=(10, 3.5))
        ax.bar(letters, d, color=["#e03131" if x < 0 else "#2f9e44" for x in d])
        ax.axhline(0, color="black", lw=0.8)
        ax.set_ylabel("Δ AP50 vs 640 FP32")
        ax.set_title(f"Effet lettre par lettre — {choice}")
        fig.tight_layout()
        fig.savefig(CHARTS / "ecart_par_lettre.png", dpi=150)
    plt.close("all")


# ---------------------------------------------------------------------------
def main():
    v = collect()
    if REFERENCE not in v:
        raise SystemExit("❌ Il faut les résultats de l'étape 4 (référence onnx_fp32_640).")
    fp, counts = dataset_fingerprint()
    choice, cands, key = decide(v)

    charts(v, choice, key)
    log_mlflow(v, choice, fp, counts)

    # Rapport texte
    col = "mAP50 216" if key == "map50_216" else "mAP50"
    lines = ["", "=" * 96, "RÉSULTATS ÉTAPE 5 — Synthèse de toutes les versions", "=" * 96,
             f"{'Version':26}{'Taille':>8}{'mAP50':>8}{col:>11}{'ms b1':>8}{'ms b8':>8}{'Perte':>8}  Règle",
             "-" * 96]
    order = sorted(v, key=lambda n: (v[n]["ms1"] or 999))
    for n in order:
        r = v[n]
        m216 = f"{r[key]:.3f}" if r.get(key) is not None else "—"
        perte = f"{r['perte']:+.3f}" if "perte" in r else "—"
        regle = "" if "ok" not in r else ("✅" if r["ok"] else "❌")
        if n == choice:
            regle += "  ← CHOISI"
        lines.append(f"{n:26}{r['taille']:>8.2f}{r['map50_test']:>8.3f}{m216:>11}"
                     f"{r['ms1']:>8.1f}{(r['ms8'] or 0):>8.1f}{perte:>8}  {regle}")
    lines += ["-" * 96,
              f"Règle : modèle INT8 le plus rapide (batch 1) avec perte ≤ {MAX_DROP:.2f} de {col} "
              f"vs {REFERENCE}",
              f"Choix : {choice}"]
    if choice and "pire_lettre" in v[choice]:
        l, d = v[choice]["pire_lettre"]
        lines.append(f"Lettre la plus dégradée pour ce choix : {l} ({d:+.3f})")
    lines += [f"Empreinte du dataset : {fp}  ({counts})",
              "⚠️  Les vitesses viennent de runs différents (±10 % de variabilité) : "
              "comparer de préférence les versions d'une même étape.",
              f"Graphiques : {CHARTS}\\", "=" * 96,
              "Interface MLflow : mlflow ui --backend-store-uri sqlite:///mlflow.db"]
    print("\n".join(lines))
    (RESULTS / "etape5_synthese.txt").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
