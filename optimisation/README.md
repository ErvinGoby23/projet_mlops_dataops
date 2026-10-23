# Optimisation — expériences et conclusions (quantization, résolution, sensibilité)

Modèle : YOLOv8n fine-tuné sur l'alphabet ASL (26 lettres, 1 512 images d'entraînement).
Mesures de vitesse : CPU AMD Ryzen 7 250 (ONNX Runtime), batch 1 = mode stream.
Précision : mAP50 sur le jeu de test (72 images) et sur le jeu élargi valid + test (216 images).

| Fichier | Question | Conclusion |
|---|---|---|
| `e1_onnx.py` | La conversion ONNX change-t-elle le modèle ? | Non : mAP identique, écart des sorties 3e-6. ONNX Runtime ×1,34 plus rapide que PyTorch |
| `e2_dynamic.py` | La quantization **dynamique** suffit-elle ? | Taille ÷3,3 sans perte, mais **×1,6 plus lente** (activations quantifiées à la volée avant chaque convolution) |
| `e3_static.py` | La quantization **statique** calibrée fait-elle mieux ? | Tout quantifier **détruit le modèle** (mAP 0) ; tête en FP32 : **aucune perte, ×1,35** |
| `e4_resolution.py` | Réduire la résolution aide-t-il ? | 416 = résolution native du dataset : **meilleure précision et ×2,8** (avec INT8) |
| `e6_sensibilite.py` | Quels blocs supportent mal l'INT8 ? Combien d'images de calibration ? | **22 blocs sur 23 tolèrent l'INT8**, seule la tête s'effondre (−0,27) ; ~100 images suffisent (4 s) |
| `e5_mlflow.py` | Quelle version déployer ? | Tout est tracé dans MLflow ; la règle de décision et le Model Registry désignent le modèle |
| `common.py` | — | Fonctions partagées (prétraitement, calibration, évaluation, benchmark) |

## Pourquoi la quantization statique « naïve » échoue (e3, variante A)

La sortie de YOLO (1 × 30 × N) mélange les **coordonnées des boîtes** (0 → 640 px)
et les **scores des lettres** (0 → 1). Avec une seule échelle `s ≈ 640/255 ≈ 2,5`,
un score de 0,9 devient `round(0,9 / 2,5) = 0` : toutes les probabilités sont écrasées.
C'est le problème des **outliers** : une plage de valeurs large étire l'échelle et détruit
la précision des petites valeurs. → On garde la tête en FP32 (**précision mixte**),
ce que l'expérience 6 confirme par la mesure.

## Recette retenue

Quantization post-training statique QDQ, MinMax, per-channel, activations uint8 / poids int8,
**opérations non-convolution de la tête (bloc 22) en FP32**, 100 images de calibration
(prises dans le jeu d'entraînement, jamais dans le test). Tous les réglages sont dans `config.yaml`.

## Résultats principaux (CPU)

| Version | Taille | mAP50 test | ms / image (b1) | Accélération |
|---|---|---|---|---|
| PyTorch FP32 640 | 6,3 Mo* | 0,924 | 45,3 | — |
| ONNX FP32 640 (référence) | 12,7 Mo | 0,924 | 36,2 | ×1,0 |
| ONNX INT8 dynamique 640 | 3,8 Mo | 0,921 | 61,0 | ×0,6 |
| ONNX INT8 statique 640 | 4,0 Mo | 0,928 | 27,5 | ×1,3 |
| **ONNX INT8 statique 416** | **3,7 Mo** | **0,939** | **12,9** | **×2,8** |
| ONNX INT8 statique 320 | 3,7 Mo | 0,934 | 8,4 | ×4,3 |

\* le `.pt` Ultralytics est stocké en FP16.

## Sur la cible (iPhone 17 Pro, CoreML, Neural Engine)

| Version | Inférence | FPS stream |
|---|---|---|
| FP32 640 | 15,0 ms | 16,6 |
| INT8 640 | 15,9 ms | 16,6 |
| FP16 416 | 12,1 ms | 16,6 |
| **INT8 416** | **12,3 ms** | 16,6 |
| INT8 320 | 9,2 ms | 16,6 |

Sur le Neural Engine, l'INT8 CoreML (poids seulement) **réduit la taille mais pas le temps** :
le calcul se fait en FP16. **L'intérêt d'une quantization dépend du matériel cible.**
Tous les modèles tiennent le temps réel (FPS plafonnés par la chaîne caméra, pas par le modèle)
→ choix final sur la précision et la taille : **INT8 416**.

## Lancer

```powershell
python run_all.py              # toutes les expériences
python run_all.py --only 6     # une seule
mlflow ui --backend-store-uri sqlite:///mlflow.db
```
