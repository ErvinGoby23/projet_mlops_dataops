# ASL Signes — alphabet de la langue des signes sur iPhone

Modèle YOLOv8n fine-tuné sur l'alphabet ASL, quantifié et embarqué dans une app iOS
native (CoreML, Neural Engine). L'inférence tourne entièrement sur le téléphone.

## Structure

| Dossier | Rôle |
|---|---|
| `model/best.pt` | Modèle entraîné (FP32, PyTorch) |
| `scripts/export_coreml.py` | Export CoreML FP32 / FP16 / INT8 + comparaison |
| `app/` | App Flutter (caméra, détection, épellation de mots) |
| `.github/workflows/ios.yml` | Build sur un Mac GitHub Actions → `.ipa` |

## Pipeline

1. Entraînement sur PC (RTX 5060) → `best.pt`
2. GitHub Actions (macOS) : export CoreML FP32 / FP16 / INT8, mesure taille / mAP / vitesse
3. La version INT8 est intégrée dans l'app, compilée sans signature → `ASL-Signes.ipa`
4. Installation sur l'iPhone avec Sideloadly (Apple ID gratuit)

## Résultats de quantization

Voir l'artefact `quantization-results` de chaque build (`benchmark.md`).

Dataset : American Sign Language Letters (Roboflow Universe, CC BY 4.0).
