# ASL Signes — détecter l'alphabet de la langue des signes sur iPhone

Reconnaissance de l'alphabet ASL **en direct et 100 % hors ligne** sur un iPhone 17 Pro.
Un YOLOv8n fine-tuné, **quantifié en INT8** puis **distillé**, tourne en CoreML sur le Neural Engine.

> Projet Master 2 EFREI — cours DevOps / DataOps (MLOps), sujet 2 « Embarquer un modèle », niveau 2 : quantifier, comparer, déployer.

**Le besoin.** Une personne apprend l'alphabet ASL seule chez elle et ne sait pas si ses gestes sont corrects.
Elle fait le signe devant le téléphone et voit tout de suite la lettre reconnue, sans réseau et sans envoyer d'image sur Internet.
Le modèle doit donc tourner **sur le téléphone** : petit, rapide et précis.

**La problématique.** Comment rendre un détecteur YOLO assez petit et rapide pour l'iPhone sans perdre en précision ?
Et le modèle choisi sur PC reste-t-il le meilleur une fois sur la cible ?

## Modèle en production

| | |
|---|---|
| Modèle | YOLOv8n **distillé** (professeur YOLOv8s), INT8, 416 px |
| Taille | **3,2 Mo** (CoreML) |
| Latence | **~12 ms** par image sur iPhone 17 Pro (Neural Engine) |
| Précision CoreML | mAP50 **0,874** (72 images de test) |
| Généralisation | **34,2 %** de bonnes lettres sur 1 300 images externes (v1 : 27,8 %) |
| Registry | MLflow `asl-detector` v7, alias `production` |

## Trois branches, une question chacune

| Branche | Question | Ce qui change | Statut |
|---|---|---|---|
| **`main`** (ce README) | Quel modèle de référence peut-on quantifier et déployer ? | YOLOv8n fine-tuné sur le dataset, INT8 statique 416, CoreML | Référence — tag `v1-dataset-seul` |
| [`distillation`](../../tree/distillation) | Un professeur améliore-t-il le petit modèle ? | Élève YOLOv8n guidé par un professeur YOLOv8s | **En production** |
| [`onnx-iphone`](../../tree/onnx-iphone) | Le modèle tourne-t-il sur iPhone sans CoreML ? | Même modèle distillé, en ONNX INT8 statique, sur le CPU | Testée sur iPhone |

## Architecture : du PC à l'iPhone, sans Mac

```
┌────────── PC (Windows, RTX 5060) ──────────┐       ┌──── GitHub Actions (macos-15) ────┐       ┌──── iPhone 17 Pro ─────┐
│ Dataset Roboflow (version 1)               │       │ scripts/export_coreml.py          │       │ App Flutter            │
│ training/train.py   → model/best.pt        │  git  │   5 variantes CoreML              │ .ipa  │ Caméra → CoreML (ANE)  │
│ optimisation/ E1 → E8  (quantization)      │──────▶│   évaluation → benchmark.csv      │──────▶│   ou ONNX Runtime (CPU)│
│ distillation/ D1 → D10 (branche dédiée)    │ push  │ flutter build ios → .ipa          │       │ Lettre puis mot épelé  │
│ MLflow : runs + Model Registry             │       │                                   │       │ 100 % hors ligne       │
└────────────────────────────────────────────┘       └───────────────────────────────────┘       └────────────────────────┘
       Expérimentation & décision                         Build automatisé (CI)                       Cible embarquée
```

Pas de Mac : la compilation iOS passe par un runner macOS GitHub, déclenché par `git push` sur `main` ou à la main.
L'app est installée par sideloading (artefact `ASL-Signes-ipa` → Sideloadly).

**Tout est versionné** : code (Git + tag), données (Roboflow version 1), modèles et mesures (MLflow), réglages (un seul `config.yaml`).

## Organisation du repo

| Chemin | Rôle |
|---|---|
| `config.yaml` | **Tous les réglages** : chemins, entraînement, calibration, résolutions, règle de décision, MLflow |
| `training/train.py` | Dataset → modèle fine-tuné (`model/best.pt`) |
| `optimisation/` | Expériences E1 → E8 — voir [`optimisation/README.md`](optimisation/README.md) |
| `run_all.py` | Enchaîne toutes les expériences en une commande |
| `scripts/export_coreml.py` | Export + quantization CoreML des 5 variantes (lancé par la CI) |
| `scripts/extraire_echantillon.py` | Échantillon d'images externes (Mendeley) pour le test de data drift |
| `app/` | Application Flutter : détection, épellation, sélecteur de modèle, mesure ms / FPS |
| `.github/workflows/ios.yml` | CI : export CoreML + compilation de l'app sur un Mac GitHub |
| `results/` | Résultats JSON + graphiques, versionnés comme preuves de l'analyse |
| `model/best.pt` | Modèle fine-tuné (point de départ de tout le reste) |

Non versionnés (`.gitignore`) : `datasets/`, `exports/`, `runs/`, `mlflow.db`, `mlruns/`.

## Données et modèle

- **Dataset** : *American Sign Language Letters* (David Lee, Roboflow Universe, version 1) — 720 photos d'origine,
  seul l'entraînement est augmenté ×3 → **1 512 entraînement / 144 validation / 72 test**.
  Images de 372 à 416 px (moyenne 393) → **416 px est la résolution native**.
- **Modèle** : YOLOv8n (~3 M paramètres) pré-entraîné sur COCO (80 classes) → fine-tuné sur 26 lettres.
  Backbone (blocs 0-9), neck (10-21), **tête (bloc 22)** — la tête est au cœur de la quantization.
- **Entraînement** : 80 epochs max, batch 16, RTX 5060, seed 0. Meilleure epoch 31 (arrêt à 51) :
  mAP50 **0,942** en validation, **0,924** en test.
- **Évaluation** : test (72 images), valid + test (216 images hors entraînement), et 1 300 images externes (Mendeley, 3 personnes).

## Pipeline d'expériences

| Étape | Rôle | Résultat |
|---|---|---|
| E1 | Export ONNX, comparaison des sorties PyTorch / ONNX | Écart 3·10⁻⁶ : base validée |
| E2 | INT8 dynamique | Rejeté : 61 ms contre 37 en FP32 |
| E3 | INT8 statique (3 variantes) | Gardé : statique, tête FP32 |
| E4 | Résolution 640 / 416 / 320 | Gardé : 416 (320 : −8 pts en CoreML) |
| E5 | MLflow : un run par version, règle de décision | Choix → Model Registry |
| E6 | Sensibilité par bloc + taille de calibration | Vérifié : seul le bloc 22 pose problème |
| E7 | Test statistique (216 images) | Vérifié : p > 0,05 |
| E8 | Test externe (1 300 images Mendeley) | Vérifié : −3 pts, Percentile retenu |

Mesures CPU : ONNX Runtime, batch 1, médiane de 3 répétitions.

## Résultats — quantization (v1)

### Variantes à 640 px (ONNX Runtime CPU, mAP50 sur 72 images)

| Version | Taille (Mo) | mAP50 | ms / image |
|---|---|---|---|
| ONNX FP32 (référence) | 12,74 | 0,924 | 37,0 |
| INT8 dynamique | 3,82 | 0,921 | 61,0 |
| INT8 statique A, tout quantifié (MinMax) | 4,02 | **0,000** | 25,4 |
| INT8 statique B, tête FP32 (MinMax) | 3,98 | 0,928 | 27,4 |
| INT8 statique C, tête FP32 (Percentile) | 3,98 | 0,921 | 27,9 |

Taille **÷3,2** (12,74 → 3,98 Mo), vitesse **×1,4** (37,0 → 27,4 ms). Le `best.pt` d'origine fait 6,3 Mo (stocké en 16 bits).

### Pourquoi tout quantifier détruit le modèle

| Blocs quantifiés seuls (416, MinMax) | Perte de mAP50 |
|---|---|
| Blocs 0 à 21 (backbone + neck) | ≤ 0,006 chacun |
| **Bloc 22 (tête)** | **−0,27** (0,939 → 0,672) |

La tête mélange de grands nombres (les boîtes, jusqu'à 640 px) et de petits nombres (les scores, entre 0 et 1).
Avec une seule échelle `s ≈ 640/255 ≈ 2,5`, un score de 0,9 devient `round(0,9 / 2,5) = 0`.
**Solution : précision mixte**, le décodage de la tête reste en FP32 (il n'a pas de poids, donc rien à alléger).

La quantization dynamique est plus lente car elle recalcule l'échelle des activations à chaque image :
elle est pensée pour les couches linéaires, pas pour les convolutions.
La calibration demande peu d'images : de 10 à 300 images, mAP50 entre 0,931 et 0,941.

### Résolution (mAP50 sur 216 images)

| Résolution | FP32 | INT8 MinMax | INT8 Percentile |
|---|---|---|---|
| 640 px | 0,932 | 0,936 | 0,933 |
| **416 px** | **0,939** | 0,936 | **0,945** |
| 320 px | 0,925 | 0,926 | 0,929 |

En INT8, la latence CPU passe de **27 → 13 → 8 ms** (640 → 416 → 320). La résolution est le levier le plus fort sur la vitesse.

### Test statistique : écart INT8 − FP32 image par image (216 images)

| Version | Bonne détection FP32 → INT8 (%) | Écart (pts) | McNemar p |
|---|---|---|---|
| 416 MinMax | 87,0 → 85,2 | −1,9 | 0,22 |
| **416 Percentile** | **87,0 → 85,6** | **−1,4** | **0,45** |
| 640 MinMax (B) | 83,3 → 81,5 | −1,9 | 0,34 |
| 640 Percentile (C) | 83,3 → 82,9 | −0,5 | 1,00 |
| 320 MinMax | 83,3 → 79,2 | −4,2 | 0,06 |
| 320 Percentile | 83,3 → 83,3 | 0,0 | 1,00 |

Juste = bonne lettre et boîte recouvrant la vraie à plus de 50 %. IC 95 % par bootstrap (10 000 tirages).
**Toutes p > 0,05 : pas de perte significative.** Percentile réduit l'écart par rapport à MinMax.

### Images d'une autre source (1 300 images Mendeley, 416 px)

| Comparaison | Bonne lettre FP32 → INT8 | Écart (pts) | McNemar p |
|---|---|---|---|
| Dataset · MinMax | 87,0 → 85,2 % | −1,9 | 0,22 |
| Dataset · Percentile | 87,0 → 85,6 % | −1,4 | 0,45 |
| Externe · MinMax | 30,9 → 26,6 % | −4,3 | < 0,001 |
| **Externe · Percentile** | **30,9 → 27,8 %** | **−3,1** | < 0,001 |

| Résolution (externe) | FP32 | MinMax | Percentile |
|---|---|---|---|
| 640 | 19,3 % | 16,5 % | 16,6 % |
| 416 | 31,2 % | 26,4 % | 28,7 % |
| 320 | 33,3 % | 26,5 % | 28,6 % |

La perte INT8 n'apparaît qu'avec le **data drift** — et le data drift coûte bien plus que l'INT8 (87 % → 31 % en FP32).

## Résultats — distillation

| Modèle (PyTorch, 640) | mAP50 test (72) | mAP50 (216) |
|---|---|---|
| v1 YOLOv8n | 0,924 | 0,932 |
| Professeur YOLOv8s | 0,917 | 0,939 |
| v1 continuée sans professeur | 0,911 | 0,909 |
| **v1 distillée** | **0,940** | **0,944** |

- **+6,4 points** de bonnes lettres sur images externes face à la v1, à taille égale (IC 95 % [+4,2 ; +8,6], p < 0,001).
- Gain positif sur **3 graines sur 3** (+7,2 pts en moyenne), significatif sur 2.
- Détails : branche [`distillation`](../../tree/distillation).

## Résultats — sur la cible

### Les 5 variantes CoreML (mesures sur Mac via la CI, mAP50 sur 72 images)

| Version | Taille (Mo) | mAP50 v1 | mAP50 distillée | ms (Mac) |
|---|---|---|---|---|
| FP16 640 (par défaut) | 6,20 | 0,804 | 0,859 | 67,8 |
| INT8 640 | 3,22 | 0,804 | 0,853 | 57,7 |
| FP16 416 | 6,17 | 0,821 | 0,854 | 31,1 |
| **INT8 416** (déployé) | **3,20** | 0,831 | **0,874** | 31,9 |
| INT8 320 | 3,20 | 0,750 | 0,744 | 19,1 |

En CoreML, l'INT8 porte sur les **poids seulement** : taille ÷1,9, mais pas plus rapide (les calculs restent en 16 bits).

### Sur l'iPhone 17 Pro (distillée INT8 416)

| Moteur | Taille | ms moy / méd | FPS |
|---|---|---|---|
| CoreML (Neural Engine) | 3,20 Mo | 11,7 / 12,2 | 15,3 |
| ONNX Runtime (CPU) | 3,77 Mo | 17,3 / 17,3 | 12,2 |

Les deux tiennent le temps réel. **L'intérêt d'une quantization dépend du matériel cible.**
Détails : branche [`onnx-iphone`](../../tree/onnx-iphone).

## Synthèse

| Retenu | Effet |
|---|---|
| Quantization INT8 statique Percentile, décodage tête en FP32 | Taille ÷3,3 (ONNX) ou ÷2 (CoreML), pas de perte significative sur le dataset (p = 0,29), −3 pts en externe |
| Résolution 416 px | ~2× plus rapide qu'en 640 (INT8, CPU) |
| Distillation (professeur v8s) | +6,4 pts en externe (p < 0,001), 3 graines sur 3 |

| Écarté | Raison |
|---|---|
| Quantization dynamique | Plus lente que le FP32 sur les convolutions |
| Tout quantifier (tête comprise) | mAP50 à 0 (640 px), en MinMax comme en Percentile |
| Résolution 320 | −8 pts en CoreML |

## Règle de décision

Suivi MLflow de chaque version (métriques, modèle, empreinte du dataset) et règle explicite dans `config.yaml` :
**perte de mAP50 ≤ 0,02** face à la référence ONNX FP32 640. Parmi les versions admissibles, on garde la plus petite et la plus rapide,
puis on vérifie sur la cible et sur des images externes.

## Reproduire

```powershell
python -m venv .venv ; .venv\Scripts\activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt

$env:ROBOFLOW_API_KEY="ta_cle"
python training/train.py                              # 1. dataset -> modèle
python run_all.py                                     # 2. expériences de quantization + MLflow
mlflow ui --backend-store-uri sqlite:///mlflow.db     #    consulter les runs et le registry
git push                                              # 3. la CI construit l'app iPhone (.ipa)
```

## Limites et pistes

- **Petits jeux d'évaluation** : 72 / 216 images ; l'externe compte 1 300 images mais seulement 3 personnes.
- **Vitesses CPU peu comparables d'un run à l'autre** (même fichier : 12,9 puis 19,1 ms) ; la comparaison se fait dans un même run.
- **Perte INT8 hors dataset** (−3 pts) : pistes QAT (quantization-aware training) et données plus variées.
- **Graines** : 3 graines pour la distillation, un seul entraînement pour les autres modèles.
- **Latence iPhone** : ms CoreML et ms ONNX définies différemment ; mAP CoreML et ONNX non comparables (NMS intégré).
- Pistes : ONNX sur le Neural Engine, mode miroir pour les gauchers, CI qui récupère le modèle directement depuis MLflow.

---

Dataset : *American Sign Language Letters* (Roboflow Universe). Images externes : dataset Mendeley.
