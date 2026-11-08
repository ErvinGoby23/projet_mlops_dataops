import 'dart:async';
import 'dart:isolate';
import 'dart:math';
import 'dart:typed_data';

import 'package:camera/camera.dart';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:onnxruntime/onnxruntime.dart';

// Taille de l'image donnee au modele (416 x 416)
const int kSize = 416;
// Les 26 lettres, dans l'ordre des classes du modele (0 = A, 25 = Z)
const String kLettres = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ';
// Confiance minimale pour afficher une lettre et sa boite
const double kSeuil = 0.25;
// Chemin du modele ONNX INT8 embarque dans l'app
const String kModele = 'assets/onnx/asl_int8_416.onnx';

// Resultat d'une prediction : la lettre, sa confiance, le temps de calcul et la boite
class Prediction {
  final String lettre;
  final double score;
  final double ms;
  final List<double> boite; // x1, y1, x2, y2 normalisés (0-1) dans l'image caméra
  const Prediction(this.lettre, this.score, this.ms, this.boite);
}

// Prepare l'image de la camera pour le modele (meme principe que Ultralytics) :
// on la reduit en 416 x 416 sans la deformer, on ajoute des bandes grises,
// on remet les couleurs en RGB et on divise par 255 pour avoir des valeurs entre 0 et 1.
Float32List letterbox(Uint8List bytes, int w, int h, int stride) {
  // Facteur de reduction : on garde les proportions de l'image
  final r = min(kSize / h, kSize / w);
  // Taille de l'image une fois reduite
  final nw = (w * r).round(), nh = (h * r).round();
  // Largeur des bandes grises a gauche et en haut (l'image est centree)
  final left = (kSize - nw) ~/ 2, top = (kSize - nh) ~/ 2;
  // Nombre de pixels d'un canal de couleur (416 x 416)
  const plan = kSize * kSize;

  // Tableau final rempli de gris (114 / 255) : les bandes gardent cette valeur
  final out = Float32List(3 * plan)..fillRange(0, 3 * plan, 114 / 255.0);
  // Pour chaque pixel de l'image reduite, on va chercher le pixel correspondant dans la camera
  for (var y = 0; y < nh; y++) {
    final sy = min(h - 1, (y / r).floor());
    for (var x = 0; x < nw; x++) {
      final sx = min(w - 1, (x / r).floor());
      // Position du pixel dans les octets de la camera (4 octets par pixel : B, G, R, A)
      final i = sy * stride + sx * 4;
      // Position du pixel dans l'image 416 x 416
      final o = (top + y) * kSize + (left + x);
      // On range canal par canal : d'abord tout le rouge, puis le vert, puis le bleu
      out[o] = bytes[i + 2] / 255.0;
      out[plan + o] = bytes[i + 1] / 255.0;
      out[2 * plan + o] = bytes[i] / 255.0;
    }
  }
  return out;
}

// Lance le modele sur une image preparee et renvoie la meilleure lettre avec sa boite
Prediction predire(OrtSession session, Float32List data, int w, int h) {
  // Entree du modele : 1 image, 3 couleurs, 416 x 416
  final input = OrtValueTensor.createTensorWithDataList(data, [1, 3, kSize, kSize]);
  final runOptions = OrtRunOptions();
  // On chronometre uniquement le calcul du modele (temps d'inference)
  final sw = Stopwatch()..start();
  final outputs = session.run(runOptions, {'images': input});
  sw.stop();
  final ms = sw.elapsedMicroseconds / 1000;

  // Sortie du modele : 30 lignes x 3549 colonnes
  // lignes 0 a 3 = boite (cx, cy, w, h), lignes 4 a 29 = score de chaque lettre
  // chaque colonne = un point de la grille du modele
  final out = (outputs[0]!.value as List)[0] as List;
  // On cherche le score le plus eleve parmi toutes les lettres et tous les points
  var best = 0.0;
  var bestCls = 0;
  var bestA = 0; // position (ancre) de la meilleure détection
  for (var c = 0; c < 26; c++) {
    final ligne = out[4 + c] as List;
    for (var a = 0; a < ligne.length; a++) {
      final s = (ligne[a] as num).toDouble();
      if (s > best) {
        best = s;
        bestCls = c;
        bestA = a;
      }
    }
  }
  // Boîte de la meilleure détection : lignes 0-3 = cx, cy, w, h en pixels de l'image 416 (letterbox)
  double v(int ligne) => ((out[ligne] as List)[bestA] as num).toDouble();
  final cx = v(0), cy = v(1), bw = v(2), bh = v(3);
  // On retire le letterbox (mêmes calculs que dans letterbox()) puis on normalise entre 0 et 1
  final r = min(kSize / h, kSize / w);
  final left = (kSize - (w * r).round()) ~/ 2;
  final top = (kSize - (h * r).round()) ~/ 2;
  // On enleve la bande grise, on annule la reduction, puis on divise par la taille de la camera
  double nx(double x) => (((x - left) / r) / w).clamp(0.0, 1.0).toDouble();
  double ny(double y) => (((y - top) / r) / h).clamp(0.0, 1.0).toDouble();
  // On passe du centre et de la taille aux 4 coins de la boite
  final boite = <double>[nx(cx - bw / 2), ny(cy - bh / 2), nx(cx + bw / 2), ny(cy + bh / 2)];

  // On libere la memoire utilisee par ONNX Runtime
  input.release();
  runOptions.release();
  for (final o in outputs) {
    o?.release();
  }
  return Prediction(kLettres[bestCls], best, ms, boite);
}

// Travail fait en arriere-plan (isolate) pour ne pas bloquer l'ecran.
// Il charge le modele une seule fois, puis traite chaque image recue.
void _travailleur(List<Object> args) {
  // Canal pour renvoyer les resultats a l'ecran
  final versUi = args[0] as SendPort;
  // Le fichier du modele, deja lu en memoire
  final modele = args[1] as Uint8List;
  // Canal pour recevoir les images
  final recu = ReceivePort();
  try {
    // Demarrage d'ONNX Runtime et chargement du modele
    OrtEnv.instance.init();
    final session = OrtSession.fromBuffer(modele, OrtSessionOptions());
    // On donne a l'ecran l'adresse ou envoyer les images
    versUi.send(recu.sendPort);
    // A chaque image recue : preparation, prediction, envoi du resultat
    recu.listen((msg) {
      try {
        final m = msg as List<Object>;
        // On mesure aussi le temps de preparation de l'image
        final sw = Stopwatch()..start();
        final entree = letterbox(m[0] as Uint8List, m[1] as int, m[2] as int, m[3] as int);
        final msPrep = sw.elapsedMicroseconds / 1000;
        final p = predire(session, entree, m[1] as int, m[2] as int);
        versUi.send(<Object>[p.lettre, p.score, p.ms, msPrep, p.boite]);
      } catch (e) {
        // En cas d'erreur sur une image, on la signale a l'ecran
        versUi.send(<Object>['ERR', '$e']);
      }
    });
  } catch (e) {
    // Erreur au chargement du modele
    versUi.send(<Object>['ERR', 'Chargement du modèle : $e']);
  }
}

// Page principale de l'app ONNX
class OnnxPage extends StatefulWidget {
  const OnnxPage({super.key});

  @override
  State<OnnxPage> createState() => _OnnxPageState();
}

class _OnnxPageState extends State<OnnxPage> {
  // Camera, isolate et canaux de communication avec l'isolate
  CameraController? _camera;
  Isolate? _isolate;
  ReceivePort? _depuisTravailleur;
  SendPort? _versTravailleur;
  String? _erreur;

  // Vrai tant qu'une image est en cours de traitement : les nouvelles sont ignorees
  bool _occupe = false;
  // Taille des images de la camera (ex : 480x640), affichee a l'ecran
  String _tailleCamera = '';

  // Derniere prediction affichee
  String? _lettre;
  double _score = 0;
  double _msInference = 0;
  double _msPreparation = 0;
  List<double>? _boite; // cadre affiché sur la caméra

  // Epellation : une lettre est ajoutee au mot si elle reste
  // a plus de 55 % de confiance pendant 900 ms
  static const _minConfiance = 0.55;
  static const _dureeTenue = Duration(milliseconds: 900);
  String? _candidat;
  DateTime _candidatDepuis = DateTime.now();
  bool _dejaAjoutee = false;
  double _progression = 0;
  String _mot = '';

  // Compteur d'images traitees par seconde (FPS)
  int _images = 0;
  double _fps = 0;
  Timer? _fpsTimer;

  // Bouton "Mesurer 10 s" : FPS, temps moyen et median
  static const _dureeMesure = 10;
  bool _mesure = false;
  int _restant = 0;
  int _imagesMesure = 0;
  final List<double> _msMesure = [];
  String? _resultatMesure;
  Timer? _timerMesure;

  // Au lancement de la page : demarrage du modele et de la camera,
  // et mise a jour du FPS toutes les secondes
  @override
  void initState() {
    super.initState();
    _demarrer();
    _fpsTimer = Timer.periodic(const Duration(seconds: 1), (_) {
      if (!mounted) return;
      setState(() {
        _fps = _images.toDouble();
        _images = 0;
      });
    });
  }

  // Charge le modele, lance l'isolate, puis ouvre la camera arriere en flux continu
  Future<void> _demarrer() async {
    try {
      // Lecture du fichier du modele embarque dans l'app
      final raw = await rootBundle.load(kModele);
      // Canal pour recevoir les resultats de l'isolate
      final port = ReceivePort();
      _depuisTravailleur = port;
      port.listen(_surReponse);
      // Lancement de l'isolate avec le modele
      _isolate = await Isolate.spawn(
          _travailleur, <Object>[port.sendPort, raw.buffer.asUint8List()]);

      // On choisit la camera arriere (sinon la premiere disponible)
      final cams = await availableCameras();
      final dos = cams.firstWhere(
        (c) => c.lensDirection == CameraLensDirection.back,
        orElse: () => cams.first,
      );
      // Camera en resolution moyenne, images au format BGRA
      final ctrl = CameraController(
        dos,
        ResolutionPreset.medium,
        imageFormatGroup: ImageFormatGroup.bgra8888,
        enableAudio: false,
      );
      await ctrl.initialize();
      // Chaque nouvelle image de la camera appelle _surImage
      await ctrl.startImageStream(_surImage);
      if (!mounted) return;
      setState(() => _camera = ctrl);
    } catch (e) {
      if (mounted) setState(() => _erreur = '$e');
    }
  }

  // Appelee pour chaque image de la camera.
  // Si une image est deja en cours de traitement, on ignore celle-ci (pas de file d'attente).
  void _surImage(CameraImage img) {
    final versTravailleur = _versTravailleur;
    if (_occupe || versTravailleur == null) return;
    _occupe = true;
    final p = img.planes[0];
    _tailleCamera = '${img.width}x${img.height}';
    // On envoie les pixels bruts, la taille et le nombre d'octets par ligne a l'isolate
    versTravailleur.send(<Object>[p.bytes, img.width, img.height, p.bytesPerRow]);
  }

  // Appelee quand l'isolate renvoie un message (son adresse, un resultat ou une erreur)
  void _surReponse(dynamic msg) {
    // Premier message : l'adresse ou envoyer les images
    if (msg is SendPort) {
      _versTravailleur = msg;
      return;
    }
    final r = msg as List<Object>;
    // Le traitement est fini : la prochaine image peut partir
    _occupe = false;
    if (r[0] == 'ERR') {
      if (mounted) setState(() => _erreur = r[1] as String);
      return;
    }
    final lettre = r[0] as String;
    final score = r[1] as double;
    final ms = r[2] as double;
    _images++;
    // Pendant une mesure, on garde le temps de chaque image
    if (_mesure) {
      _imagesMesure++;
      _msMesure.add(ms);
    }

    // Epellation : la lettre doit etre assez sure et rester la meme pendant 900 ms
    final candidat = score >= _minConfiance ? lettre : null;
    final maintenant = DateTime.now();
    // Si la lettre change, on recommence le compte a zero
    if (candidat != _candidat) {
      _candidat = candidat;
      _candidatDepuis = maintenant;
      _dejaAjoutee = false;
    }
    final tenue = maintenant.difference(_candidatDepuis);
    double progression = 0;
    if (candidat != null && !_dejaAjoutee) {
      // Avancement du cercle autour de la lettre (de 0 a 1)
      progression = (tenue.inMilliseconds / _dureeTenue.inMilliseconds).clamp(0.0, 1.0);
      // Lettre tenue assez longtemps : on l'ajoute au mot et le telephone vibre
      if (tenue >= _dureeTenue) {
        _mot += candidat;
        _dejaAjoutee = true;
        progression = 0;
        HapticFeedback.mediumImpact();
      }
    }

    // Mise a jour de l'ecran
    if (!mounted) return;
    setState(() {
      _msInference = ms;
      _msPreparation = r[3] as double;
      _score = score;
      // On n'affiche la lettre et la boite que si la confiance depasse 25 %
      _lettre = score >= kSeuil ? lettre : null;
      _boite = score >= kSeuil
          ? (r[4] as List).map((e) => (e as num).toDouble()).toList()
          : null;
      _progression = progression;
    });
  }

  // Bouton "Mesurer 10 s" : compte les images pendant 10 secondes,
  // puis calcule le FPS, le temps moyen et le temps median d'inference
  void _lancerMesure() {
    if (_mesure) return;
    setState(() {
      _mesure = true;
      _imagesMesure = 0;
      _msMesure.clear();
      _restant = _dureeMesure;
    });
    _timerMesure = Timer.periodic(const Duration(seconds: 1), (t) {
      if (!mounted) return;
      setState(() => _restant--);
      if (_restant <= 0) {
        t.cancel();
        // FPS = nombre d'images traitees / 10 secondes
        final fps = _imagesMesure / _dureeMesure;
        var texte = '${fps.toStringAsFixed(1)} FPS | inférence n/a';
        if (_msMesure.isNotEmpty) {
          // Moyenne et mediane des temps d'inference
          final tri = [..._msMesure]..sort();
          final moy = _msMesure.reduce((a, b) => a + b) / _msMesure.length;
          final med = tri[tri.length ~/ 2];
          texte = '${fps.toStringAsFixed(1)} FPS | inférence moy '
              '${moy.toStringAsFixed(1)} ms | méd ${med.toStringAsFixed(1)} ms';
        }
        setState(() {
          _resultatMesure = 'ONNX INT8 416 (CPU) : $texte';
          _mesure = false;
        });
        HapticFeedback.heavyImpact();
      }
    });
  }

  // A la fermeture de la page : on arrete les minuteurs, la camera et l'isolate
  @override
  void dispose() {
    _fpsTimer?.cancel();
    _timerMesure?.cancel();
    _camera?.stopImageStream();
    _camera?.dispose();
    _isolate?.kill(priority: Isolate.immediate);
    _depuisTravailleur?.close();
    super.dispose();
  }

  // Cadre noir semi-transparent utilise pour les panneaux du haut et du bas
  Widget _panneau({required Widget child}) => Container(
        margin: const EdgeInsets.symmetric(horizontal: 12, vertical: 6),
        padding: const EdgeInsets.all(12),
        decoration: BoxDecoration(
          color: Colors.black.withValues(alpha: 0.65),
          borderRadius: BorderRadius.circular(16),
        ),
        child: child,
      );

  // Construction de l'ecran : apercu camera, panneau d'infos en haut, mot en bas
  @override
  Widget build(BuildContext context) {
    final cam = _camera;
    final scheme = Theme.of(context).colorScheme;
    return Scaffold(
      backgroundColor: Colors.black,
      body: Stack(
        fit: StackFit.expand,
        children: [
          // Apercu de la camera avec la boite dessinee par-dessus
          if (cam != null && cam.value.isInitialized)
            Center(
              child: CameraPreview(
                cam,
                // Le child de CameraPreview a exactement la taille de l'aperçu
                child: CustomPaint(painter: BoitePainter(_boite, _lettre, scheme.primary)),
              ),
            )
          else
            // Rond de chargement tant que la camera n'est pas prete
            const Center(child: CircularProgressIndicator()),
          // Panneau du haut : lettre, confiance, temps, FPS et bouton de mesure
          SafeArea(
            child: Align(
              alignment: Alignment.topCenter,
              child: _panneau(
                child: Column(
                  mainAxisSize: MainAxisSize.min,
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Row(mainAxisSize: MainAxisSize.min, children: [
                      // Lettre reconnue entouree du cercle de progression
                      SizedBox(
                        width: 56,
                        height: 56,
                        child: Stack(alignment: Alignment.center, children: [
                          CircularProgressIndicator(
                            value: _progression,
                            strokeWidth: 4,
                            color: scheme.primary,
                            backgroundColor: Colors.white12,
                          ),
                          Text(_lettre ?? '-',
                              style: const TextStyle(fontSize: 30, fontWeight: FontWeight.bold)),
                        ]),
                      ),
                      const SizedBox(width: 14),
                      // Informations : modele, confiance, temps d'inference et de preparation, FPS
                      Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
                        const Text('Modèle : ONNX INT8 416 (CPU)'),
                        Text('Confiance : ${(_score * 100).toStringAsFixed(0)} %'),
                        Text('Inférence : ${_msInference.toStringAsFixed(1)} ms',
                            style: const TextStyle(color: Colors.white70)),
                        Text('Préparation : ${_msPreparation.toStringAsFixed(1)} ms',
                            style: const TextStyle(color: Colors.white70)),
                        Text('${_fps.toStringAsFixed(0)} FPS · caméra $_tailleCamera',
                            style: const TextStyle(color: Colors.white70)),
                      ]),
                    ]),
                    const SizedBox(height: 8),
                    // Bouton de mesure sur 10 secondes
                    FilledButton(
                      onPressed: _mesure ? null : _lancerMesure,
                      child: Text(_mesure ? 'Mesure… $_restant s' : 'Mesurer 10 s'),
                    ),
                    // Resultat de la derniere mesure
                    if (_resultatMesure != null)
                      Padding(
                        padding: const EdgeInsets.only(top: 8),
                        child: SelectableText(_resultatMesure!),
                      ),
                    // Message d'erreur eventuel
                    if (_erreur != null)
                      Padding(
                        padding: const EdgeInsets.only(top: 8),
                        child: Text('Erreur : $_erreur',
                            style: const TextStyle(color: Colors.redAccent)),
                      ),
                  ],
                ),
              ),
            ),
          ),
          // Panneau du bas : le mot epele et les boutons Espace, Retour, Effacer
          SafeArea(
            child: Align(
              alignment: Alignment.bottomCenter,
              child: _panneau(
                child: Column(
                  mainAxisSize: MainAxisSize.min,
                  children: [
                    Text(
                      _mot.isEmpty ? 'Tiens une lettre ~1 s pour l\'écrire' : _mot,
                      textAlign: TextAlign.center,
                      style: TextStyle(
                        fontSize: _mot.isEmpty ? 15 : 32,
                        fontWeight: FontWeight.w600,
                        letterSpacing: 2,
                        color: _mot.isEmpty ? Colors.white60 : Colors.white,
                      ),
                    ),
                    const SizedBox(height: 10),
                    Row(children: [
                      // Ajoute un espace au mot
                      Expanded(
                          child: FilledButton.tonal(
                              onPressed: () => setState(() => _mot += ' '),
                              child: const Text('Espace'))),
                      const SizedBox(width: 8),
                      // Supprime la derniere lettre
                      Expanded(
                          child: FilledButton.tonal(
                              onPressed: _mot.isEmpty
                                  ? null
                                  : () => setState(
                                      () => _mot = _mot.substring(0, _mot.length - 1)),
                              child: const Text('Retour'))),
                      const SizedBox(width: 8),
                      // Efface tout le mot
                      Expanded(
                          child: FilledButton(
                              onPressed: () => setState(() => _mot = ''),
                              child: const Text('Effacer'))),
                    ]),
                  ],
                ),
              ),
            ),
          ),
        ],
      ),
    );
  }
}
/// Dessine le cadre de la détection et la lettre au-dessus de l'aperçu caméra.
class BoitePainter extends CustomPainter {
  final List<double>? boite;
  final String? lettre;
  final Color couleur;
  BoitePainter(this.boite, this.lettre, this.couleur);

  // Dessine la boite et la lettre sur l'apercu camera
  @override
  void paint(Canvas canvas, Size size) {
    final b = boite;
    if (b == null) return;
    // Les coordonnees sont entre 0 et 1 : on les multiplie par la taille de l'apercu
    final rect = Rect.fromLTRB(
        b[0] * size.width, b[1] * size.height, b[2] * size.width, b[3] * size.height);
    // Cadre de la boite
    canvas.drawRect(
        rect,
        Paint()
          ..style = PaintingStyle.stroke
          ..strokeWidth = 3
          ..color = couleur);
    // Etiquette avec la lettre, placee juste au-dessus du cadre
    if (lettre != null) {
      final tp = TextPainter(
        text: TextSpan(
          text: ' $lettre ',
          style: TextStyle(
              color: Colors.white,
              fontSize: 20,
              fontWeight: FontWeight.bold,
              backgroundColor: couleur),
        ),
        textDirection: TextDirection.ltr,
      )..layout();
      tp.paint(canvas, Offset(rect.left, max(0.0, rect.top - tp.height)));
    }
  }

  // On redessine seulement si la boite ou la lettre a change
  @override
  bool shouldRepaint(BoitePainter old) => old.boite != boite || old.lettre != lettre;
}