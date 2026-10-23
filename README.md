# ASL Signes — alphabet de la langue des signes, embarqué sur iPhone

Reconnaissance de l'alphabet ASL en temps réel, **entièrement sur le téléphone** (hors ligne).
Modèle YOLOv8n fine-tuné, **quantifié** et déployé en CoreML sur le Neural Engine d'un iPhone 17 Pro.
Usage : l'interlocuteur filme la personne qui épelle un mot ; l'app affiche les lettres et le mot.

## Architecture

```
┌──────── PC (RTX 5060) ────────┐     ┌──── GitHub Actions (macOS) ────┐     ┌──── iPhone 17 Pro ────┐
│ training/train.py             │     │ scripts/export_coreml.py       │     │ app/ (Flutter)        │
│   dataset Roboflow → best.pt  │     │   5 versions CoreML            │     │   caméra (stream)     │
│ optimisation/ (e1 → e6)       │ git │   FP32/INT8 640, FP16/INT8 416,│ .ipa│   CoreML sur ANE      │
│   ONNX, PTQ dynamique/statique│────▶│   INT8 320 + évaluation        │────▶│   lettre + mot épelé  │
│   résolution, sensibilité     │push │ Compilation Flutter → .ipa     │     │   mesure ms / FPS     │
│ MLflow (runs + registry)      │     │                                │     │   100 % hors ligne    │
└───────────────────────────────┘     └────────────────────────────────┘     └───────────────────────┘
   Expérimentation & analyse              Build automatisé (CI)                   Cible embarquée
```

## Organisation du repo

| Chemin | Rôle |
|---|---|
| `config.yaml` | **Tous les réglages** (chemins, résolutions, calibration, règle de décision, MLflow) |
| `training/train.py` | Données → modèle (`model/best.pt`) |
| `optimisation/` | Les 6 expériences d'optimisation (quantization, résolution, sensibilité) + `common.py` (code partagé) — voir son README |
| `run_all.py` | Relance tout le pipeline de quantization en une commande |
| `scripts/export_coreml.py` | Export et quantization CoreML pour l'iPhone (lancé par la CI) |
| `app/` | Application iPhone (Flutter) : détection, épellation, sélecteur de modèle, mesures |
| `.github/workflows/ios.yml` | CI : export CoreML + compilation de l'app sur un Mac GitHub |
| `results/` | Résultats JSON + graphiques (versionnés : preuves de l'analyse) |
| `model/best.pt` | Modèle entraîné (point de départ) |

Non versionnés (`.gitignore`) : `datasets/`, `exports/`, `runs/`, `mlflow.db`, `mlruns/`.

## Reproduire

```powershell
python -m venv .venv ; .venv\Scripts\activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt

$env:ROBOFLOW_API_KEY="ta_cle"
python training/train.py        # 1. données -> modèle
python run_all.py               # 2. expériences de quantization + MLflow
mlflow ui --backend-store-uri sqlite:///mlflow.db
git push                        # 3. la CI construit l'app iPhone (.ipa)
```

Installation sur iPhone : télécharger l'artefact `ASL-Signes-ipa` du build GitHub, puis Sideloadly.

## Résultats clés

- **Quantization statique INT8 + précision mixte** (tête en FP32) : taille ÷3,2, ×1,35 plus rapide sur CPU, **sans perte**.
- **Résolution 416** (native du dataset) : ×2,8 avec INT8, meilleure précision.
- **Sensibilité** : 22 blocs sur 23 tolèrent l'INT8 ; toute la perte vient de la tête de détection.
- **Sur l'iPhone** : 12 ms par image ; l'INT8 CoreML réduit la taille mais pas le temps (calcul en FP16 sur le Neural Engine).

Détails : [`optimisation/README.md`](optimisation/README.md).

Dataset : *American Sign Language Letters* (Roboflow Universe, CC BY 4.0).
