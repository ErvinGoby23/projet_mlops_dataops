# Branche `onnx-iphone` — le même modèle sur iPhone, sans CoreML

Troisième volet du projet [ASL Signes](../../tree/main).
**Question** : le modèle tourne-t-il sur l'iPhone **sans CoreML**, avec la quantization ONNX complète (poids + activations) ?
**Réponse** : oui, en temps réel — **~17 ms** sur le CPU avec ONNX Runtime, contre **~12 ms** sur le Neural Engine avec CoreML.

## Ce qui change par rapport à `main`

| | CoreML (branches `main` / `distillation`) | ONNX (cette branche) |
|---|---|---|
| Modèle | YOLOv8n distillé, graine 0, 416 px | **Le même** |
| Quantization | INT8 sur les **poids seulement** | INT8 statique **poids + activations** (Percentile, tête FP32) |
| Fichier | `.mlpackage` — 3,20 Mo | `app/assets/onnx/asl_int8_416.onnx` — 3,77 Mo |
| Moteur | Neural Engine (plugin `ultralytics_yolo`) | CPU (package Flutter `onnxruntime`) |
| Pré / post-traitement | Intégré au plugin (NMS inclus) | Écrit à la main en Dart (`onnx_page.dart`) |

Seuls la quantization et le moteur changent : c'est exactement le modèle validé sur PC (expérience D7) qui tourne sur le téléphone.

## L'app ONNX (`app/lib/onnx_page.dart`)

1. **Caméra** en flux (BGRA) → **letterbox 416×416** (fond gris 114, conservation du ratio) → tenseur RGB normalisé.
2. **Inférence** `session.run` dans un **isolate** Dart séparé : l'interface reste fluide, plus de lag.
3. **Décodage** de la sortie YOLO : meilleure lettre au-dessus du seuil 0,25, boîte remise dans le repère caméra.
4. **Affichage** : lettre, boîte, **épellation** du mot, et bouton « Mesurer 10 s » (ms moyenne / médiane, FPS).

Cette app s'ouvre directement sur la page ONNX ; les mesures CoreML viennent de l'app de `main` / `distillation`, sur le même téléphone.

## Résultats

### Le modèle embarqué, mesuré sur PC (ONNX Runtime CPU, 416 px)

| Version | Taille | mAP50 (216) | Bonne lettre dataset | Bonne lettre externe |
|---|---|---|---|---|
| Distillée FP32 | 12,53 Mo | 0,932 | 88,0 % | 37,3 % |
| **Distillée INT8 Percentile** (embarquée) | **3,77 Mo** | **0,924** | **86,1 %** | **34,2 %** |
| v1 INT8 Percentile (pour comparaison) | 3,7 Mo | 0,945 | 85,6 % | 27,8 % |

Taille **÷3,3**. Pas de perte significative sur le dataset (McNemar p = 0,29) ; −3,1 pts sur images externes (p < 0,001).

### Mesures sur iPhone 17 Pro

Distillée INT8 416, « Mesurer 10 s », chaque moteur mesuré 2 fois.

| Moteur · mesure | Taille | ms moy / méd | FPS |
|---|---|---|---|
| CoreML (Neural Engine) · 1 | 3,20 Mo | 11,7 / 12,2 | 15,3 |
| CoreML (Neural Engine) · 2 | 3,20 Mo | 12,4 / 12,4 | 15,0 |
| ONNX Runtime (CPU) · 1 | 3,77 Mo | 17,3 / 17,3 | 12,2 |
| ONNX Runtime (CPU) · 2 | 3,77 Mo | 17,4 / 17,3 | 12,0 |

| Synthèse | CoreML | ONNX |
|---|---|---|
| Latence médiane | ~12 ms | ~17 ms |
| FPS | ~15 | ~12 |
| Taille | 3,20 Mo | 3,77 Mo |
| Ce qui est quantifié | Poids | Poids + activations |
| Matériel | Neural Engine | CPU |

- Le **Neural Engine est plus rapide** que le CPU (environ 1,4×) ; les deux tiennent le temps réel.
- Les mesures sont stables : moins de 1 ms d'écart entre deux mesures du même moteur.
- **Définitions différentes** : ms ONNX = calcul seul (`session.run`) ; ms CoreML = chiffre renvoyé par le plugin YOLO.
  Les FPS dépendent de la lumière et de la chaîne caméra : la ms est la mesure la plus stable.
- La mAP CoreML et la mAP ONNX ne sont pas directement comparables (NMS intégré côté CoreML).

**Conclusion** : la quantization ONNX complète est portable sur iPhone sans dépendre d'Apple, mais
**sur ce matériel CoreML reste le meilleur choix** : le modèle déployé reste la version CoreML.
Piste : faire tourner ONNX sur le Neural Engine (execution provider CoreML d'ONNX Runtime).

## Build (CI)

`.github/workflows/ios.yml` se déclenche sur un `push` vers `onnx-iphone` :

1. vérifie la présence du modèle `app/assets/onnx/asl_int8_416.onnx` et affiche son empreinte SHA-256 ;
2. génère le projet iOS et compile en release sans signature (`flutter build ios --no-codesign`) ;
3. publie l'artefact **`ASL-ONNX-ipa`**, à installer par sideloading (Sideloadly).

Pour remplacer le modèle : exporter l'INT8 416 Percentile depuis la branche [`distillation`](../../tree/distillation) (D7),
le copier dans `app/assets/onnx/asl_int8_416.onnx`, puis pousser.
