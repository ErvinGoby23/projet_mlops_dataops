# Branche `distillation` — un professeur pour le petit modèle

Deuxième optimisation du projet ASL Signes (branche `main`) : un **YOLOv8s** (professeur) guide le **YOLOv8n** (élève)
pendant l'entraînement. L'élève garde **sa taille et sa vitesse** : seule la précision peut changer.

**Question** : un professeur améliore-t-il le petit modèle, et le gain survit-il à la quantization et au passage sur iPhone ?
**Réponse** : oui pour la **généralisation** (+6,4 points sur images nouvelles), pas pour le score sur le dataset.
C'est **le modèle en production**, en **ONNX INT8 416** sur le CPU de l'iPhone (MLflow `asl-detector` v7, alias `production`).

> La v1 (`model/best.pt`, tag `v1-dataset-seul`) n'est pas modifiée : tous les modèles de cette branche sont écrits dans de nouveaux fichiers.

## Principe

```
                 Professeur YOLOv8s (11,1 M, 22,5 Mo) — d1
                              │ guide (imitation des features, dis = 6,0)
                              ▼
 Point de départ : v1 YOLOv8n (3,0 M, best.pt)
        ├── d2 · v1 + professeur         → v1 distillée   (perte de distillation 8,1 → 1,8)
        └── d3 · v1 continuée seule      → contrôle       (isole l'effet du professeur)
```

| Rôle      | Modèle                                | Paramètres | Fichier                             |
| ---------- | -------------------------------------- | ----------- | ----------------------------------- |
| Professeur | YOLOv8s, entraîné sur le dataset ASL | 11,1 M      | `model/teacher_v8s.pt` (22,5 Mo)  |
| Élève    | YOLOv8n = la v1 comme point de départ | 3,0 M       | `model/best_distill.pt` (6,26 Mo) |

Même départ, même recette, même environnement (Ultralytics, GPU RTX 5060) : **seule différence = le professeur**.
Seul l'élève est déployé. Export ONNX vérifié : écart PyTorch/ONNX < 4·10⁻⁴ px sur les boîtes, 10⁻⁷ sur les scores.

## Scripts

| Script                    | Question                                                         | Conclusion                                                                               |
| ------------------------- | ---------------------------------------------------------------- | ---------------------------------------------------------------------------------------- |
| `d1_professeur.py`      | Un modèle 3,6× plus gros fait-il mieux que la v1 ?             | **Non** : aussi bon, pas meilleur → la limite vient des données                  |
| `d2_eleve.py`           | La v1 distillée s'améliore-t-elle ?                            | Oui à 640 : 0,932 → **0,944** mAP50 (216 images)                                 |
| `d3_controle.py`        | Le gain vient-il du professeur ou d'un entraînement plus long ? | **Du professeur** : sans lui la v1 se dégrade (effet : +0,035 mAP50)              |
| `d4_quantif_distill.py` | Le gain survit-il à l'INT8 à 416 ?                             | Taille ÷3,3, pas de perte significative sur le dataset                                  |
| `d5_robustesse.py`      | Test statistique INT8 vs FP32                                    | Même méthode que E7 : bootstrap IC 95 % + McNemar                                      |
| `d6_externe.py`         | Et sur des images d'une autre source ?                           | **+6,4 pts** face à la v1 INT8 (p < 0,001)                                        |
| `d7_percentile.py`      | MinMax ou Percentile ?                                           | Égalité sur le dataset, Percentile perd 1,2 pt de moins en externe → **retenu** |
| `d8_resolutions.py`     | 640 / 416 / 320 pour la distillée                               | 416 reste le meilleur compromis                                                          |
| `d9_graines.py`         | Le gain dépend-il de la graine ?                                | Positif sur **3 graines sur 3** (+7,2 pts en moyenne)                              |
| `d10_mlflow.py`         | Quelle version déployer ?                                       | Distillée INT8 416 → registry `asl-detector` v7, `production`                      |

## Résultats

### D1 — Professeur vs v1 (PyTorch, 640 px)

| Modèle            | Taille   | mAP50 test (72) | mAP50-95 test | mAP50 (216) | mAP50-95 (216) |
| ------------------ | -------- | --------------- | ------------- | ----------- | -------------- |
| v1 YOLOv8n         | 6,26 Mo  | 0,924           | 0,765         | 0,932       | 0,776          |
| Professeur YOLOv8s | 22,54 Mo | 0,917           | 0,746         | 0,939       | 0,765          |

Professeur : arrêt précoce à l'epoch 41, meilleure epoch 21. 3,6× plus gros, pas nettement meilleur : **la limite vient des données**.

### D2 + D3 — Distillation et contrôle (PyTorch, 640 px)

| Modèle                                      | mAP50 test (72) | mAP50-95 test   | mAP50 (216)     | mAP50-95 (216)  |
| -------------------------------------------- | --------------- | --------------- | --------------- | --------------- |
| v1                                           | 0,924           | 0,765           | 0,932           | 0,776           |
| v1 continuée **sans** professeur (d3) | 0,911           | 0,774           | 0,909           | 0,761           |
| **v1 distillée** (d2)                 | **0,940** | **0,778** | **0,944** | **0,780** |

- Distillée : 49 epochs (18 min), meilleure epoch 29. Contrôle : 29 epochs, meilleure epoch 9.
- Effet du professeur (distillée − contrôle) : **+0,035 mAP50** sur 216 images.
- Interprétation : le professeur n'en sait pas plus que la v1, mais ses prédictions « nuancées » servent de
  **régularisation** et empêchent l'élève de se dégrader pendant le nouvel entraînement.

### D4 + D7 — Distillée INT8 (ONNX Runtime CPU, 416 px, statique, tête FP32, 100 images de calibration)

| Version                                   | Taille            | mAP50 (216)     | Bonne lettre (externe) |
| ----------------------------------------- | ----------------- | --------------- | ---------------------- |
| FP32 (non quantifiée)                    | 12,53 Mo          | 0,932           | 37,3 %                 |
| INT8 MinMax (D4)                          | 3,77 Mo           | 0,927           | 33,0 %                 |
| **INT8 Percentile (D7)** — retenue | **3,77 Mo** | **0,924** | **34,2 %**       |

### D5 — Test statistique INT8 vs FP32

| Jeu                    | Bonne lettre FP32 → INT8 | Écart (pts) | IC 95 %         | McNemar p |
| ---------------------- | ------------------------- | ------------ | --------------- | --------- |
| Dataset (216 images)   | 88,0 → 86,1 %            | −1,9        | [−4,6 ; +0,5]  | 0,29      |
| Externe (1 300 images) | 37,3 → 34,2 %            | −3,1        | [−4,5 ; −1,6] | < 0,001   |

Pas de perte significative sur le dataset ; perte réelle sur images externes (piste : QAT).
Vitesse ~12,9 ms sur CPU, comme la v1 (même architecture). Taille **÷3,3** (12,53 → 3,77 Mo).

### D6 — Sur des images nouvelles (1 300 images Mendeley, 416 px)

| Comparaison                        | Résultat                             | IC 95 %       | McNemar p |
| ---------------------------------- | ------------------------------------- | ------------- | --------- |
| Distillée INT8 vs v1 INT8         | **34,2 % vs 27,8 % (+6,4 pts)** | [+4,2 ; +8,6] | < 0,001   |
| Distillée FP32 vs professeur FP32 | 37,3 % vs 34,8 %                      | —            | 0,054     |

L'élève **égale son professeur** pour un modèle 3,6× plus petit.

### D9 — Robustesse sur 3 graines (INT8 416, 1 300 images externes)

Chaque graine entraîne une distillée et son contrôle sans professeur.

|                                      | Valeur                                          |
| ------------------------------------ | ----------------------------------------------- |
| Gain moyen (distillée − contrôle) | **+7,2 pts**                              |
| Plage                                | +1,8 à +13,4 pts                               |
| Graines avec gain positif            | 3 sur 3                                         |
| McNemar                              | p < 0,001 (graines 0 et 2), p = 0,13 (graine 1) |

La graine 0 (déployée) est la meilleure des trois : le gain réaliste est d'environ +7 pts, pas +13.

### CoreML — format iPhone (mAP50 sur 72 images, CI macOS)

| Version                | v1    | Distillée      | Écart           |
| ---------------------- | ----- | --------------- | ---------------- |
| FP16 640 (par défaut) | 0,804 | 0,859           | +0,055           |
| INT8 640               | 0,804 | 0,853           | +0,049           |
| FP16 416               | 0,821 | 0,854           | +0,033           |
| **INT8 416**     | 0,831 | **0,874** | **+0,043** |
| INT8 320               | 0,750 | 0,744           | −0,006          |

La distillée est devant dans 4 versions sur 5. Même taille (3,20 Mo) et même vitesse que la v1 : ~12 ms par image sur iPhone 17 Pro en CoreML.
En production, c'est la version **ONNX INT8 416** qui tourne sur l'iPhone : 3,77 Mo, ~17 ms par image (CPU), 12,2 FPS — détails dans la branche `onnx-iphone`.

**Leçon : le professeur améliore la généralisation, pas le score sur le dataset.**

## Lancer

```powershell
python distillation/d1_professeur.py     # professeur YOLOv8s -> model/teacher_v8s.pt
python distillation/d2_eleve.py          # v1 distillée       -> model/best_distill.pt
python distillation/d3_controle.py       # contrôle sans professeur
python distillation/d4_quantif_distill.py
# d5 à d9 : tests statistiques, externe, calibration, résolutions, graines
python distillation/d10_mlflow.py        # synthèse + registry
```

Prérequis et installation : voir le README de la branche `main`.
`model/teacher_v8s.pt` (22,5 Mo) n'est pas versionné : il est régénéré par `d1`
