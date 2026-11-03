import 'dart:async';
import 'dart:isolate';
import 'dart:math';
import 'dart:typed_data';

import 'package:camera/camera.dart';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:onnxruntime/onnxruntime.dart';

/// Détection ASL avec le modèle ONNX INT8 416 (distillée, Percentile),
/// exécuté par ONNX Runtime sur le CPU de l'iPhone, dans un fil séparé (isolate)
/// pour ne pas bloquer l'affichage de la caméra.
const int kSize = 416;
const String kLettres = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ';
const double kSeuil = 0.25; // affichage : même seuil que e7 / d6
const String kModele = 'assets/onnx/asl_int8_416.onnx';

class Prediction {
  final String lettre;
  final double score;
  final double ms;
  const Prediction(this.lettre, this.score, this.ms);
}

/// Même chose que letterbox() de common.py : sans déformer, bandes grises 114,
/// RGB, /255, format 1x3x416x416. "rotation" redresse l'image de la caméra.
Float32List letterbox(Uint8List bytes, int w, int h, int stride, int rotation) {
  final tourne = rotation == 90 || rotation == 270;
  final uw = tourne ? h : w; // taille de l'image redressée
  final uh = tourne ? w : h;
  final r = min(kSize / uh, kSize / uw);
  final nw = (uw * r).round(), nh = (uh * r).round();
  final left = (kSize - nw) ~/ 2, top = (kSize - nh) ~/ 2;
  const plan = kSize * kSize;

  final out = Float32List(3 * plan)..fillRange(0, 3 * plan, 114 / 255.0);
  for (var y = 0; y < nh; y++) {
    final uy = min(uh - 1, (y / r).floor());
    for (var x = 0; x < nw; x++) {
      final ux = min(uw - 1, (x / r).floor());
      int sx, sy;
      if (rotation == 90) {
        sx = uy;
        sy = h - 1 - ux;
      } else if (rotation == 180) {
        sx = w - 1 - ux;
        sy = h - 1 - uy;
      } else if (rotation == 270) {
        sx = w - 1 - uy;
        sy = ux;
      } else {
        sx = ux;
        sy = uy;
      }
      final i = sy * stride + sx * 4; // BGRA
      final o = (top + y) * kSize + (left + x);
      out[o] = bytes[i + 2] / 255.0; // R
      out[plan + o] = bytes[i + 1] / 255.0; // G
      out[2 * plan + o] = bytes[i] / 255.0; // B
    }
  }
  return out;
}

/// Lance le modèle et garde la détection la plus confiante (comme top1() de e7 / d6).
Prediction predire(OrtSession session, Float32List data) {
  final input = OrtValueTensor.createTensorWithDataList(data, [1, 3, kSize, kSize]);
  final runOptions = OrtRunOptions();
  final sw = Stopwatch()..start();
  final outputs = session.run(runOptions, {'images': input});
  sw.stop();
  final ms = sw.elapsedMicroseconds / 1000;

  final out = (outputs[0]!.value as List)[0] as List; // [30][3549]
  var best = 0.0;
  var bestCls = 0;
  for (var c = 0; c < 26; c++) {
    final ligne = out[4 + c] as List;
    for (var a = 0; a < ligne.length; a++) {
      final s = (ligne[a] as num).toDouble();
      if (s > best) {
        best = s;
        bestCls = c;
      }
    }
  }
  input.release();
  runOptions.release();
  for (final o in outputs) {
    o?.release();
  }
  return Prediction(kLettres[bestCls], best, ms);
}

/// Le fil séparé : charge le modèle une fois, puis traite chaque image reçue.
void _travailleur(List<Object> args) {
  final versUi = args[0] as SendPort;
  final modele = args[1] as Uint8List;
  final recu = ReceivePort();
  try {
    OrtEnv.instance.init();
    final session = OrtSession.fromBuffer(modele, OrtSessionOptions());
    versUi.send(recu.sendPort);
    recu.listen((msg) {
      try {
        final m = msg as List<Object>;
        final sw = Stopwatch()..start();
        final entree = letterbox(m[0] as Uint8List, m[1] as int, m[2] as int,
            m[3] as int, m[4] as int);
        final msPrep = sw.elapsedMicroseconds / 1000;
        final p = predire(session, entree);
        versUi.send(<Object>[p.lettre, p.score, p.ms, msPrep]);
      } catch (e) {
        versUi.send(<Object>['ERR', '$e']);
      }
    });
  } catch (e) {
    versUi.send(<Object>['ERR', 'Chargement du modèle : $e']);
  }
}

class OnnxPage extends StatefulWidget {
  const OnnxPage({super.key});

  @override
  State<OnnxPage> createState() => _OnnxPageState();
}

class _OnnxPageState extends State<OnnxPage> {
  CameraController? _camera;
  Isolate? _isolate;
  ReceivePort? _depuisTravailleur;
  SendPort? _versTravailleur;
  String? _erreur;

  bool _occupe = false;
  int _rotation = 0; // l'image arrive déjà en portrait (480x640) ; ajustable avec le bouton
  String _tailleCamera = '';

  String? _lettre;
  double _score = 0;
  double _msInference = 0;
  double _msPreparation = 0;

  // Épellation (comme l'app CoreML) : une lettre est validée si elle reste stable assez longtemps
  static const _minConfiance = 0.55;
  static const _dureeTenue = Duration(milliseconds: 900);
  String? _candidat;
  DateTime _candidatDepuis = DateTime.now();
  bool _dejaAjoutee = false;
  double _progression = 0;
  String _mot = '';

  int _images = 0;
  double _fps = 0;
  Timer? _fpsTimer;

  static const _dureeMesure = 10;
  bool _mesure = false;
  int _restant = 0;
  int _imagesMesure = 0;
  final List<double> _msMesure = [];
  String? _resultatMesure;
  Timer? _timerMesure;

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

  Future<void> _demarrer() async {
    try {
      // 1. Le fil séparé qui fait tout le calcul
      final raw = await rootBundle.load(kModele);
      final port = ReceivePort();
      _depuisTravailleur = port;
      port.listen(_surReponse);
      _isolate = await Isolate.spawn(
          _travailleur, <Object>[port.sendPort, raw.buffer.asUint8List()]);

      // 2. La caméra
      final cams = await availableCameras();
      final dos = cams.firstWhere(
        (c) => c.lensDirection == CameraLensDirection.back,
        orElse: () => cams.first,
      );
      final ctrl = CameraController(
        dos,
        ResolutionPreset.medium,
        imageFormatGroup: ImageFormatGroup.bgra8888,
        enableAudio: false,
      );
      await ctrl.initialize();
      await ctrl.startImageStream(_surImage);
      if (!mounted) return;
      setState(() => _camera = ctrl);
    } catch (e) {
      if (mounted) setState(() => _erreur = '$e');
    }
  }

  /// Image caméra : on l'envoie au fil séparé, sauf s'il est encore occupé.
  void _surImage(CameraImage img) {
    final versTravailleur = _versTravailleur;
    if (_occupe || versTravailleur == null) return; // on saute les images pendant un calcul
    _occupe = true;
    final p = img.planes[0];
    _tailleCamera = '${img.width}x${img.height}';
    versTravailleur.send(<Object>[p.bytes, img.width, img.height, p.bytesPerRow, _rotation]);
  }

  /// Réponse du fil séparé.
  void _surReponse(dynamic msg) {
    if (msg is SendPort) {
      _versTravailleur = msg; // le modèle est chargé
      return;
    }
    final r = msg as List<Object>;
    _occupe = false;
    if (r[0] == 'ERR') {
      if (mounted) setState(() => _erreur = r[1] as String);
      return;
    }
    final lettre = r[0] as String;
    final score = r[1] as double;
    final ms = r[2] as double;
    _images++;
    if (_mesure) {
      _imagesMesure++;
      _msMesure.add(ms);
    }

    // Épellation : la lettre doit rester la même pendant 0,9 s
    final candidat = score >= _minConfiance ? lettre : null;
    final maintenant = DateTime.now();
    if (candidat != _candidat) {
      _candidat = candidat;
      _candidatDepuis = maintenant;
      _dejaAjoutee = false;
    }
    final tenue = maintenant.difference(_candidatDepuis);
    double progression = 0;
    if (candidat != null && !_dejaAjoutee) {
      progression = (tenue.inMilliseconds / _dureeTenue.inMilliseconds).clamp(0.0, 1.0);
      if (tenue >= _dureeTenue) {
        _mot += candidat;
        _dejaAjoutee = true;
        progression = 0;
        HapticFeedback.mediumImpact();
      }
    }

    if (!mounted) return;
    setState(() {
      _msInference = ms;
      _msPreparation = r[3] as double;
      _score = score;
      _lettre = score >= kSeuil ? lettre : null;
      _progression = progression;
    });
  }

  void _tourner() => setState(() => _rotation = (_rotation + 90) % 360);

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
        final fps = _imagesMesure / _dureeMesure;
        var texte = '${fps.toStringAsFixed(1)} FPS | inférence n/a';
        if (_msMesure.isNotEmpty) {
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

  Widget _panneau({required Widget child}) => Container(
        margin: const EdgeInsets.symmetric(horizontal: 12, vertical: 6),
        padding: const EdgeInsets.all(12),
        decoration: BoxDecoration(
          color: Colors.black.withValues(alpha: 0.65),
          borderRadius: BorderRadius.circular(16),
        ),
        child: child,
      );

  @override
  Widget build(BuildContext context) {
    final cam = _camera;
    final scheme = Theme.of(context).colorScheme;
    return Scaffold(
      backgroundColor: Colors.black,
      body: Stack(
        fit: StackFit.expand,
        children: [
          if (cam != null && cam.value.isInitialized)
            Center(child: CameraPreview(cam))
          else
            const Center(child: CircularProgressIndicator()),

          // ---- Haut : lettre + chiffres + boutons ----
          SafeArea(
            child: Align(
              alignment: Alignment.topCenter,
              child: _panneau(
                child: Column(
                  mainAxisSize: MainAxisSize.min,
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Row(mainAxisSize: MainAxisSize.min, children: [
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
                          Text(_lettre ?? '–',
                              style: const TextStyle(fontSize: 30, fontWeight: FontWeight.bold)),
                        ]),
                      ),
                      const SizedBox(width: 14),
                      Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
                        const Text('Modèle : ONNX INT8 416 (CPU)'),
                        Text('Confiance : ${(_score * 100).toStringAsFixed(0)} %'),
                        Text('Inférence : ${_msInference.toStringAsFixed(1)} ms',
                            style: const TextStyle(color: Colors.white70)),
                        Text('Préparation : ${_msPreparation.toStringAsFixed(1)} ms',
                            style: const TextStyle(color: Colors.white70)),
                        Text('${_fps.toStringAsFixed(0)} FPS · caméra $_tailleCamera · rotation $_rotation°',
                            style: const TextStyle(color: Colors.white70)),
                      ]),
                    ]),
                    const SizedBox(height: 8),
                    Row(mainAxisSize: MainAxisSize.min, children: [
                      OutlinedButton(onPressed: _tourner, child: Text('Rotation $_rotation°')),
                      const SizedBox(width: 8),
                      FilledButton(
                        onPressed: _mesure ? null : _lancerMesure,
                        child: Text(_mesure ? 'Mesure… $_restant s' : 'Mesurer 10 s'),
                      ),
                    ]),
                    if (_resultatMesure != null)
                      Padding(
                        padding: const EdgeInsets.only(top: 8),
                        child: SelectableText(_resultatMesure!),
                      ),
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

          // ---- Bas : mot épelé ----
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
                      Expanded(
                          child: FilledButton.tonal(
                              onPressed: () => setState(() => _mot += ' '),
                              child: const Text('Espace'))),
                      const SizedBox(width: 8),
                      Expanded(
                          child: FilledButton.tonal(
                              onPressed: _mot.isEmpty
                                  ? null
                                  : () => setState(
                                      () => _mot = _mot.substring(0, _mot.length - 1)),
                              child: const Text('⌫'))),
                      const SizedBox(width: 8),
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