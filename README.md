# Distillation — expériences et conclusion (branche `distillation`)

Deuxième optimisation tentée (sujet : « 1 à 3 optimisations, quantization, distillation… »).
**La v1 n'est pas modifiée** : tous les modèles de cette partie sont écrits dans de nouveaux fichiers.

## Principe

Un **gros modèle** (le *professeur*) aide un **petit modèle** (l'*élève*) pendant son entraînement :
l'élève apprend les bonnes réponses du dataset **et** imite les prédictions du professeur.
L'élève garde sa taille et sa vitesse : **seule la précision peut changer**.
Outil : distillation intégrée à Ultralytics (`distill_model=…`, poids `dis` = 6,0 par défaut).

| Rôle | Modèle | Paramètres | Fichier |
|---|---|---|---|
| Professeur | YOLOv8s, entraîné sur le dataset ASL | 11,1 M | `model/teacher_v8s.pt` (22,5 Mo) |
| Élève | YOLOv8n = **la v1** (`model/best.pt`) comme point de départ | 3,0 M | `model/best_distill.pt` (6,3 Mo) |

## Fichiers

| Fichier | Question | Conclusion |
|---|---|---|
| `d1_professeur.py` | Un modèle 3,7× plus gros fait-il mieux que la v1 ? | **Non** : aussi bon, pas meilleur → la limite vient des données |
| `d2_eleve.py` | La v1 distillée par le professeur s'améliore-t-elle ? | Oui à 640 : +0,012 mAP50 (216 images) |
| `d3_controle.py` | Le gain vient-il du professeur ou d'un entraînement plus long ? | **Du professeur** : sans lui, la v1 se dégrade (effet du professeur : +0,035 mAP50) |
| `d4_quantif_distill.py` | Le gain survit-il à la quantization INT8, à 416 (résolution déployée) ? | **Non** : à 416, la distillée est un peu moins bonne → **v1 INT8 416 conservée** |

## Résultats (mesurés, CPU)

### D1 — Professeur vs v1 (PyTorch, 640)

| Modèle | Taille | mAP50 test (72) | mAP50-95 test | mAP50 (216) | mAP50-95 (216) |
|---|---|---|---|---|---|
| v1 YOLOv8n | 6,26 Mo | 0,924 | 0,765 | 0,932 | 0,776 |
| Professeur YOLOv8s | 22,54 Mo | 0,917 | 0,746 | 0,939 | 0,765 |

Professeur : arrêt précoce epoch 41, meilleure epoch 21.

### D2 + D3 — Distillation et contrôle (PyTorch, 640)

| Modèle | mAP50 test (72) | mAP50-95 test | mAP50 (216) | mAP50-95 (216) |
|---|---|---|---|---|
| v1 | 0,924 | 0,765 | 0,932 | 0,776 |
| v1 continuée **sans** professeur (d3) | 0,911 | 0,774 | 0,909 | 0,761 |
| **v1 distillée** (d2) | **0,940** | **0,778** | **0,944** | **0,780** |

- Distillée : 49 epochs (18 min), meilleure epoch 29. Contrôle : 29 epochs, meilleure epoch 9.
- Même départ (la v1), même recette, même environnement (Ultralytics 8.4.174, GPU) : **seule différence = le professeur**.
- Interprétation (déduite) : le professeur n'en sait pas plus que la v1, mais ses prédictions « nuancées » servent de
  **régularisation** et empêchent l'élève de se dégrader pendant ce nouvel entraînement.

### D4 — Distillation + quantization (ONNX Runtime CPU, 416)

| Modèle | Taille | mAP50 test (72) | mAP50-95 test | mAP50 (216) | mAP50-95 (216) | ms/img b1 |
|---|---|---|---|---|---|---|
| v1 FP32 | 12,50 Mo | 0,935 | 0,786 | 0,939 | 0,796 | 22,0 |
| **v1 INT8 (production)** | **3,73 Mo** | **0,939** | **0,814** | **0,936** | **0,809** | **19,1** |
| distillée FP32 | 12,53 Mo | 0,928 | 0,774 | 0,932 | 0,784 | 22,8 |
| distillée INT8 | 3,77 Mo | 0,913 | 0,777 | 0,927 | 0,792 | 19,3 |

Distillée INT8 − v1 INT8 (216 images) : **mAP50 −0,010 | mAP50-95 −0,017**.
La recette de quantization (tête FP32) fonctionne aussi sur le modèle distillé (taille ÷3,3, sans perte).
⚠️ Vitesses mesurées dans un autre run que le README d'`optimisation/` : ne comparer qu'à l'intérieur de ce tableau.

## Décision

**La v1 INT8 416 reste en production.** La distillation améliore la v1 à 640 (gain prouvé par le contrôle),
mais pas à 416, la résolution déployée. Comme pour le 320 : **la décision se prend dans les conditions de déploiement**.

## Limites

- Une seule graine par entraînement : variabilité d'un entraînement à l'autre non mesurée.
- Petits écarts (1 à 2 points) ; jeu de test de 72 images (1 à 5 par lettre).
- Les « 216 images » incluent la validation, qui sert à choisir la meilleure epoch (vrai pour tous les modèles).
- Professeur et élève entraînés à 640 ; une distillation directement à 416 n'a pas été testée.

## Lancer

```powershell
# venv GPU (entraînements) : ..\..\projet\.venv\Scripts\Activate.ps1
python distillation/d1_professeur.py     # professeur + comparaison à la v1
python distillation/d2_eleve.py          # v1 distillée
python distillation/d3_controle.py       # contrôle sans professeur

# venv CPU (ONNX, quantization) : .venv\Scripts\Activate.ps1
python distillation/d4_quantif_distill.py
```

Chaque script saute l'entraînement si son modèle existe déjà. Résultats : `results/distill_d1.json` → `distill_d4.json`.
