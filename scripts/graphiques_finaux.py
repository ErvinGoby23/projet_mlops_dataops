"""
Graphiques finaux pour l'oral (résultats du 02/11/2026).
Les valeurs sont recopiées des sorties des scripts (d6, d9, CI CoreML) : voir le doc « Récap optimisations ».
Lancer depuis la racine du repo :  python graphiques_finaux.py
Les images sont enregistrées dans results/graphiques/.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

OUT = Path("results/graphiques")
OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})

BLEU, ORANGE, GRIS = "#2E6FD8", "#F08A24", "#9AA0A6"


def etiquettes(ax, barres, fmt="{:.1f}"):
    for b in barres:
        ax.annotate(fmt.format(b.get_height()), (b.get_x() + b.get_width() / 2, b.get_height()),
                    ha="center", va="bottom", fontsize=9, xytext=(0, 2), textcoords="offset points")


# ---------- 1. Les 3 modèles sur les 1 300 images externes (d6, confirmation, 416) ----------
modeles = ["v1\n(YOLOv8n)", "Distillée\n(YOLOv8n + prof)", "v8m\n(gros modèle)"]
fp32 = [30.9, 37.3, 36.9]
int8 = [27.8, 34.2, 34.7]
taille_int8 = ["3,7 Mo", "3,8 Mo", "26 Mo"]

x = np.arange(len(modeles))
fig, ax = plt.subplots(figsize=(8, 5))
b1 = ax.bar(x - 0.2, fp32, 0.4, label="FP32 (non quantifié)", color=GRIS)
b2 = ax.bar(x + 0.2, int8, 0.4, label="INT8 (quantifié)", color=BLEU)
etiquettes(ax, b1)
etiquettes(ax, b2)
for i, t in enumerate(taille_int8):
    ax.text(x[i] + 0.2, int8[i] + 3.2, t, ha="center", fontsize=10, fontweight="bold", color=BLEU)
ax.set_xticks(x, modeles)
ax.set_ylabel("Bonne lettre (%)")
ax.set_ylim(0, 45)
ax.set_title("1 300 images d'une autre source (jamais vues)")
ax.legend(loc="upper left")
fig.tight_layout()
fig.savefig(OUT / "externe_3_modeles.png", dpi=200)

# ---------- 2. Graines : distillée vs contrôle sans professeur (d9, INT8 416) ----------
graines = ["Graine 0", "Graine 1", "Graine 2"]
distill = [34.2, 30.3, 30.7]
controle = [20.8, 28.5, 24.4]

x = np.arange(len(graines))
fig, ax = plt.subplots(figsize=(8, 5))
b1 = ax.bar(x - 0.2, controle, 0.4, label="Sans professeur (contrôle)", color=GRIS)
b2 = ax.bar(x + 0.2, distill, 0.4, label="Avec professeur (distillée)", color=BLEU)
etiquettes(ax, b1)
etiquettes(ax, b2)
ax.axhline(np.mean(distill), color=BLEU, ls="--", lw=1)
ax.axhline(np.mean(controle), color=GRIS, ls="--", lw=1)
ax.text(2.55, np.mean(distill), f"moy. {np.mean(distill):.1f}", color=BLEU, va="center", fontsize=9)
ax.text(2.55, np.mean(controle), f"moy. {np.mean(controle):.1f}", color=GRIS, va="center", fontsize=9)
ax.set_xticks(x, graines)
ax.set_ylabel("Bonne lettre, INT8 (%)")
ax.set_ylim(0, 40)
ax.set_xlim(-0.6, 2.9)
ax.set_title("La distillée gagne à chaque graine (+7 pts en moyenne)")
ax.legend(loc="upper right")
fig.tight_layout()
fig.savefig(OUT / "graines_distillation.png", dpi=200)

# ---------- 3. CoreML : v1 vs distillée (CI, test 72 images) ----------
versions = ["FP32 640", "INT8 640", "FP16 416", "INT8 416", "INT8 320"]
v1 = [0.804, 0.804, 0.821, 0.831, 0.750]
dist = [0.859, 0.853, 0.854, 0.874, 0.744]

x = np.arange(len(versions))
fig, ax = plt.subplots(figsize=(9, 5))
b1 = ax.bar(x - 0.2, v1, 0.4, label="v1", color=GRIS)
b2 = ax.bar(x + 0.2, dist, 0.4, label="Distillée", color=ORANGE)
etiquettes(ax, b1, "{:.3f}")
etiquettes(ax, b2, "{:.3f}")
ax.set_xticks(x, versions)
ax.set_ylabel("mAP50 (CoreML, 72 images)")
ax.set_ylim(0.6, 0.92)
ax.set_title("Format iPhone (CoreML) : la distillée tient")
ax.legend(loc="upper left")
fig.tight_layout()
fig.savefig(OUT / "coreml_v1_vs_distillee.png", dpi=200)

# ---------- 4. Coût de la quantization : dataset vs images externes ----------
modeles = ["v1", "Distillée", "v8m"]
perte_dataset = [-1.4, -1.9, 0.0]   # e7 / d5, 216 images jamais vues (v1 et distillée Percentile, v8m MinMax)
perte_externe = [-3.1, -3.1, -2.2]  # d6, confirmation, 1 300 images

x = np.arange(len(modeles))
fig, ax = plt.subplots(figsize=(8, 5))
b1 = ax.bar(x - 0.2, perte_dataset, 0.4, label="Images du dataset (216)", color=GRIS)
b2 = ax.bar(x + 0.2, perte_externe, 0.4, label="Images d'une autre source (1 300)", color=BLEU)
for b in list(b1) + list(b2):
    h = b.get_height()
    ax.annotate(f"{h:+.1f}", (b.get_x() + b.get_width() / 2, h), ha="center",
                va="top" if h < 0 else "bottom", fontsize=9, xytext=(0, -3 if h < 0 else 2),
                textcoords="offset points")
ax.axhline(0, color="black", lw=0.8)
ax.set_xticks(x, modeles)
ax.set_ylabel("Écart INT8 − FP32 (points)")
ax.set_ylim(-4.5, 0.8)
ax.set_title("La quantization ne coûte qu'avec le data drift")
ax.legend(loc="lower left")
fig.tight_layout()
fig.savefig(OUT / "cout_quantization_dataset_vs_externe.png", dpi=200)

# ---------- 5. Résolution de la distillée : dataset vs externe (d8, INT8 Percentile) ----------
res = ["640", "416", "320"]
map50_dataset = [0.937, 0.924, 0.917]
externe = [23.7, 35.9, 36.2]  # premier échantillon, 1 300 images

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5))
b = ax1.bar(res, map50_dataset, color=GRIS)
etiquettes(ax1, b, "{:.3f}")
ax1.set_ylim(0.85, 0.96)
ax1.set_title("Dataset d'origine (mAP50, 216 img)")
ax1.set_xlabel("Résolution (px)")
b = ax2.bar(res, externe, color=[GRIS, BLEU, GRIS])
etiquettes(ax2, b)
ax2.set_ylim(0, 45)
ax2.set_title("Images externes (bonne lettre %, 1 300 img)")
ax2.set_xlabel("Résolution (px)")
fig.suptitle("Le 640 gagne sur le dataset mais perd 12 pts sur des images nouvelles")
fig.tight_layout()
fig.savefig(OUT / "resolution_distillee.png", dpi=200)

# ---------- 6. Taille vs précision sur images externes (confirmation, 416) ----------
points = [  # (nom, taille Mo, bonne lettre %, couleur, marqueur)
    ("v1 INT8 (production)", 3.73, 27.8, GRIS, "o"),
    ("Distillée INT8", 3.77, 34.2, BLEU, "o"),
    ("v8m INT8", 26.0, 34.7, ORANGE, "o"),
    ("v1 FP32", 12.5, 30.9, GRIS, "s"),
    ("Distillée FP32", 12.5, 37.3, BLEU, "s"),
    ("Professeur v8s FP32", 42.9, 34.8, "#7B61C9", "s"),
    ("v8m FP32", 99.0, 36.9, ORANGE, "s"),
]
fig, ax = plt.subplots(figsize=(8.5, 5))
for nom, t, acc, c, m in points:
    ax.scatter(t, acc, s=90, color=c, marker=m, zorder=3)
    ax.annotate(nom, (t, acc), xytext=(6, 4), textcoords="offset points", fontsize=9)
ax.set_xscale("log")
ax.set_xticks([3, 10, 30, 100], ["3 Mo", "10 Mo", "30 Mo", "100 Mo"])
ax.set_xlim(2.5, 200)
ax.set_ylim(25, 40)
ax.set_xlabel("Taille du modèle ONNX (échelle log)")
ax.set_ylabel("Bonne lettre sur images externes (%)")
ax.set_title("Meilleur compromis : la distillée INT8 (en haut à gauche)")
ax.grid(alpha=0.3)
ax.text(2.6, 25.5, "● INT8   ■ FP32", fontsize=9)
fig.tight_layout()
fig.savefig(OUT / "taille_vs_precision_externe.png", dpi=200)

print("✅ 6 graphiques enregistrés dans", OUT)