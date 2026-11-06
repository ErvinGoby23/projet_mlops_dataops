"""
Experience 5 — MLflow : tracer toutes les versions + graphiques + regle de decision

Principe MLOps : "ML system source code = training code + data" -> versionnement conjoint
du DATASET (empreinte SHA256), des MODELES (.onnx/.pt) et des METRIQUES.

Deroulement :
  1. Lecture des resultats des experiences 1 a 4
  2. Creation d'un run MLflow par version (parametres, metriques, artefacts)
  3. Application de la regle de decision automatique (definie dans config.yaml)
  4. Enregistrement du modele selectionne dans le Model Registry (alias "production")
  5. Generation des graphiques comparatifs pour le rapport final

Interface MLflow : mlflow ui --backend-store-uri sqlite:///mlflow.db -> http://127.0.0.1:5000
"""

import hashlib
import json
import sys
from pathlib import Path

# Ajout du dossier parent au chemin systeme pour les importations internes
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use("Agg")  # Configuration de Matplotlib en mode non interactif
import matplotlib.pyplot as plt

from optimisation.common import CFG, P, ROOT

# Repertoires et parametres globaux extraits de la configuration
RESULTS = P["results"]
CHARTS = RESULTS / "graphiques"
EXPORTS = P["exports"]
DATA_DIR = P["data"].parent
MAX_DROP = CFG["decision"]["max_drop_map50"]
REFERENCE = CFG["decision"]["reference"]
TRACKING_URI = CFG["mlflow"]["tracking_uri"]
EXPERIMENT = CFG["mlflow"]["experiment"]
REGISTERED_NAME = CFG["mlflow"]["registered_model"]


# ---------------------------------------------------------------------------
# FONCTIONS UTILITAIRES & COLLECTE DES DONNEES
# ---------------------------------------------------------------------------

def load(n):
    """Charge le fichier JSON de resultats d'une etape donnee s'il existe."""
    p = RESULTS / f"etape{n}.json"
    return json.loads(p.read_text()) if p.exists() else None


def ms(d, bs):
    """Extrait le temps en millisecondes d'un dictionnaire de benchmark (gestion des cles str/int)."""
    if not d:
        return None
    return d.get(str(bs), d.get(bs))


def collect():
    """Rassemble l'ensemble des resultats des etapes 1 a 4 dans une structure unifiee."""
    v = {}
    e1, e2, e3, e4 = load(1), load(2), load(3), load(4)

    # Experience 1 : PyTorch FP32
    if e1:
        p = e1["pytorch"]
        v["pytorch_fp32_640"] = dict(
            etape=1, format="PyTorch", precision="FP32*", methode="aucune",
            imgsz=640, taille=p["taille_mo"], map50_test=p["mAP50"],
            map5095_test=p["mAP50-95"], ms1=ms(p["ms_par_image"], 1),
            ms8=ms(p["ms_par_image"], 8), fichier=P["model_pt"]
        )
        
    # Experience 2 : ONNX INT8 Dynamique
    if e2:
        d = e2["onnx_int8_dynamique"]
        v["onnx_int8_dyn_640"] = dict(
            etape=2, format="ONNX", precision="INT8", methode="PTQ dynamique",
            imgsz=640, taille=d["taille_mo"], map50_test=d["mAP50"],
            map5095_test=d["mAP50-95"], ms1=ms(d["ms_par_image"], 1),
            ms8=ms(d["ms_par_image"], 8),
            fichier=EXPORTS / "asl_int8_dyn_640.onnx"
        )
        
    # Experience 3 : ONNX INT8 Statique (Variantes A et C à 640px)
    if e3:
        for key, label, meth in [
            ("A_minmax_tout", "onnx_int8_static_A_640", "PTQ statique MinMax, tout quantifie"),
            ("C_percentile_tete_fp32", "onnx_int8_static_C_640", "PTQ statique Percentile, tete FP32")
        ]:
            if key in e3:
                d = e3[key]
                v[label] = dict(
                    etape=3, format="ONNX", precision="INT8", methode=meth, imgsz=640,
                    taille=d["taille_mo"], map50_test=d["mAP50_test"],
                    map5095_test=d["mAP50-95_test"], map50_216=d["mAP50_216img"],
                    ms1=ms(d["ms_par_image"], 1), ms8=ms(d["ms_par_image"], 8),
                    lettres=d.get("ap50_par_lettre"),
                    fichier=EXPORTS / f"asl_int8_static_{key}.onnx"
                )
                
    # Experience 4 : Multi-resolutions (640, 416, 320) x (FP32, INT8)
    if e4:
        for size in (640, 416, 320):
            for prec in ("FP32", "INT8"):
                d = e4.get(f"{size}_{prec}")
                if not d:
                    continue
                name = f"onnx_{prec.lower()}_{size}" if prec == "FP32" else f"onnx_int8_static_{size}"
                v[name] = dict(
                    etape=4, format="ONNX", precision=prec,
                    methode="aucune" if prec == "FP32" else f"PTQ statique {CFG['quantization'].get('calib_method', 'minmax').capitalize()}, tete FP32",
                    imgsz=size, taille=d["taille_mo"], map50_test=d["mAP50_test"],
                    map5095_test=d["mAP50-95_test"], map50_216=d["mAP50_216img"],
                    ms1=ms(d["ms_par_image"], 1), ms8=ms(d["ms_par_image"], 8),
                    lettres=d.get("ap50_par_lettre"),
                    fichier=EXPORTS / (f"asl_fp32_{size}.onnx" if prec == "FP32" else f"asl_int8_static_{size}.onnx")
                )
    return v


def dataset_fingerprint():
    """Calcule le hash SHA256 unique des fichiers du dataset pour la tracabilite MLOps."""
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
# SELECTION ET TRAÇABILITÉ MLFLOW
# ---------------------------------------------------------------------------

def decide(v):
    """Applique la regle de decision : modele INT8 le plus rapide avec perte <= seuil tolere."""
    ref = v[REFERENCE]
    key = "map50_216" if "map50_216" in ref else "map50_test"
    cands = []
    
    for name, r in v.items():
        if r["precision"] != "INT8" or r.get(key) is None or r.get("ms1") is None:
            continue
        r["perte"] = r[key] - ref[key]
        r["ok"] = r["perte"] >= -MAX_DROP
        if r.get("lettres") and ref.get("lettres"):
            worst = min(
                ((l, r["lettres"].get(l, 0) - ap) for l, ap in ref["lettres"].items()),
                key=lambda x: x[1]
            )
            r["pire_lettre"] = worst
        cands.append(name)
        
    ok = [n for n in cands if v[n]["ok"]]
    choice = min(ok, key=lambda n: v[n]["ms1"]) if ok else None
    return choice, cands, key


def log_mlflow(v, choice, fp, counts):
    """Enregistre l'ensemble des runs, metriques et artefacts dans le serveur MLflow."""
    try:
        import mlflow
        from mlflow.tracking import MlflowClient
    except ImportError:
        print("Attention: mlflow non installe (pip install mlflow)")
        return
        
    uri = TRACKING_URI
    if uri.startswith("sqlite:///") and not Path(uri[10:]).is_absolute():
        uri = "sqlite:///" + str((ROOT / uri[10:]).as_posix())
        
    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment(EXPERIMENT)
    
    # Creation d'un run par variante testee
    for name, r in v.items():
        with mlflow.start_run(run_name=name) as run:
            mlflow.set_tags({
                "etape": r["etape"],
                "dataset_empreinte": fp,
                "choisi": str(name == choice)
            })
            mlflow.log_params({
                "format": r["format"],
                "precision": r["precision"],
                "methode": r["methode"],
                "imgsz": r["imgsz"],
                "dataset_train": counts["train"],
                "dataset_valid": counts["valid"],
                "dataset_test": counts["test"]
            })
            metrics = {
                "taille_mo": r["taille"],
                "mAP50_test": r["map50_test"],
                "mAP50-95_test": r["map5095_test"],
                "ms_image_batch1": r["ms1"],
                "ms_image_batch8": r["ms8"],
                "mAP50_216img": r.get("map50_216"),
                "perte_mAP50": r.get("perte")
            }
            mlflow.log_metrics({k: float(x) for k, x in metrics.items() if x is not None})

            # Publication dans le Model Registry du modele selectionne
            f = Path(r["fichier"])
            if name == choice and f.suffix == ".onnx" and f.exists():
                try:
                    import onnx
                    import mlflow.onnx
                    try:
                        info = mlflow.onnx.log_model(
                            onnx.load(str(f)), name="model", registered_model_name=REGISTERED_NAME
                        )
                    except TypeError:
                        info = mlflow.onnx.log_model(
                            onnx.load(str(f)), artifact_path="model", registered_model_name=REGISTERED_NAME
                        )
                    client = MlflowClient()
                    version = getattr(info, "registered_model_version", None) or max(
                        int(m.version) for m in client.search_model_versions(f"name='{REGISTERED_NAME}'")
                    )
                    client.set_registered_model_alias(REGISTERED_NAME, "production", str(version))
                    print(f"OK: {name} enregistre -> models:/{REGISTERED_NAME}@production (v{version})")
                except Exception as e:
                    print(f"Attention: Enregistrement dans le registry impossible ({e})")
                    mlflow.log_artifact(str(f), artifact_path="model")
            elif f.exists():
                mlflow.log_artifact(str(f), artifact_path="model")
        print(f"  Run MLflow journalise : {name}")

    # Run de synthese globale
    with mlflow.start_run(run_name="synthese"):
        mlflow.set_tags({"dataset_empreinte": fp, "choisi": choice or "aucun"})
        e6 = RESULTS / "etape6.json"
        if e6.exists():
            d = json.loads(e6.read_text(encoding="utf-8"))
            worst = min(d["blocs"].items(), key=lambda kv: kv[1]["perte_mAP50"])
            mlflow.log_metrics({
                "bloc_le_plus_sensible": float(worst[0]),
                "perte_bloc_le_plus_sensible": float(worst[1]["perte_mAP50"])
            })
            mlflow.log_artifact(str(e6))
        if CHARTS.exists():
            mlflow.log_artifacts(str(CHARTS), artifact_path="graphiques")
        mlflow.log_artifact(str(ROOT / "config.yaml"))


# ---------------------------------------------------------------------------
# GÉNÉRATION DES GRAPHIQUES
# ---------------------------------------------------------------------------

def charts(v, choice, key):
    """Genere les visuels comparatifs (Vitesse vs Precision, Taille, Degradation par lettre)."""
    CHARTS.mkdir(parents=True, exist_ok=True)
    pts = {n: r for n, r in v.items() if r.get("ms1") and r.get(key, r["map50_test"]) is not None}

    # 1. Graphique : Vitesse vs Precision (taille des bulles = poids Mo)
    fig, ax = plt.subplots(figsize=(8, 5))
    for n, r in pts.items():
        y = r.get(key) if r.get(key) is not None else r["map50_test"]
        color = "#d9480f" if n == choice else ("#1c7ed6" if r["precision"] == "INT8" else "#868e96")
        ax.scatter(r["ms1"], y, s=r["taille"] * 25, color=color, alpha=0.8)
        ax.annotate(
            n.replace("onnx_", ""), (r["ms1"], y), fontsize=8, xytext=(5, 4), textcoords="offset points"
        )
    ref = v[REFERENCE]
    ax.axhline(ref[key] - MAX_DROP, color="#e03131", ls="--", lw=1, label=f"seuil (-{MAX_DROP:.2f})")
    ax.set_xlabel("ms par image, batch 1 (CPU) — plus a gauche = plus rapide")
    ax.set_ylabel("mAP50" + (" (216 images)" if key == "map50_216" else " (test)"))
    ax.set_title("Compromis vitesse / precision (taille du point = taille du modele)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(CHARTS / "vitesse_vs_precision.png", dpi=150)

    # 2. Graphique : Comparaison des tailles de fichier
    fig, ax = plt.subplots(figsize=(8, 4))
    names = list(v)
    ax.barh(
        [n.replace("onnx_", "") for n in names],
        [v[n]["taille"] for n in names],
        color=["#d9480f" if n == choice else "#1c7ed6" for n in names]
    )
    ax.set_xlabel("Taille du fichier (Mo)")
    ax.set_title("Taille des modeles")
    fig.tight_layout()
    fig.savefig(CHARTS / "taille.png", dpi=150)

    # 3. Graphique : Ecart par lettre pour le modele retenu
    if choice and v[choice].get("lettres") and ref.get("lettres"):
        letters = sorted(ref["lettres"])
        d = [v[choice]["lettres"].get(l, 0) - ref["lettres"][l] for l in letters]
        fig, ax = plt.subplots(figsize=(10, 3.5))
        ax.bar(letters, d, color=["#e03131" if x < 0 else "#2f9e44" for x in d])
        ax.axhline(0, color="black", lw=0.8)
        ax.set_ylabel("Delta AP50 vs 640 FP32")
        ax.set_title(f"Effet lettre par lettre — {choice}")
        fig.tight_layout()
        fig.savefig(CHARTS / "ecart_par_lettre.png", dpi=150)
        
    plt.close("all")


# ---------------------------------------------------------------------------
# MAIN & CONSTITUTION DU RAPPORT TEXTE
# ---------------------------------------------------------------------------

def main():
    v = collect()
    if REFERENCE not in v:
        raise SystemExit("Erreur: Résultats manquants pour l'etape 4 (reference onnx_fp32_640 requise).")
        
    fp, counts = dataset_fingerprint()
    choice, cands, key = decide(v)

    # Generation des visuels et sauvegarde MLflow
    charts(v, choice, key)
    log_mlflow(v, choice, fp, counts)

    # Construction du tableau et du rapport texte console/fichier
    col = "mAP50 216" if key == "map50_216" else "mAP50"
    lines = [
        "",
        "=" * 96,
        "RESULTATS ETAPE 5 — Synthese de toutes les versions",
        "=" * 96,
        f"{'Version':26}{'Taille':>8}{'mAP50':>8}{col:>11}{'ms b1':>8}{'ms b8':>8}{'Perte':>8}  Regle",
        "-" * 96
    ]
    
    order = sorted(v, key=lambda n: (v[n]["ms1"] or 999))
    for n in order:
        r = v[n]
        m216 = f"{r[key]:.3f}" if r.get(key) is not None else "—"
        perte = f"{r['perte']:+.3f}" if "perte" in r else "—"
        regle = "" if "ok" not in r else ("OK" if r["ok"] else "NOK")
        if n == choice:
            regle += "  <- CHOISI"
        lines.append(
            f"{n:26}{r['taille']:>8.2f}{r['map50_test']:>8.3f}{m216:>11}"
            f"{r['ms1']:>8.1f}{(r['ms8'] or 0):>8.1f}{perte:>8}  {regle}"
        )
        
    lines += [
        "-" * 96,
        f"Regle : modele INT8 le plus rapide (batch 1) avec perte <= {MAX_DROP:.2f} de {col} vs {REFERENCE}",
        f"Choix : {choice}"
    ]
    
    if choice and "pire_lettre" in v[choice]:
        l, d = v[choice]["pire_lettre"]
        lines.append(f"Lettre la plus degradee pour ce choix : {l} ({d:+.3f})")
        
    lines += [
        f"Empreinte du dataset : {fp}  ({counts})",
        "Remarque: Les vitesses proviennent de runs differents (+/- 10% de variabilite).",
        f"Graphiques generes : {CHARTS}",
        "=" * 96,
        "Interface MLflow : mlflow ui --backend-store-uri sqlite:///mlflow.db"
    ]
    
    # Output console et enregistrement dans etape5_synthese.txt
    print("\n".join(lines))
    (RESULTS / "etape5_synthese.txt").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()