import 'dart:async';
import 'dart:isolate';
import 'dart:math';
import 'dart:typed_data';

import 'package:camera/camera.dart';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:onnxruntime/onnxruntime.dart';

const int kSize = 416;
const String kLettres = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ';
const double kSeuil = 0.25;
const String kModele = 'assets/onnx/asl_int8_416.onnx';

class Prediction {
  final String lettre;
  final double score;
  final double ms;
  const Prediction(this.lettre, this.score, this.ms);
}

Float32List letterbox(Uint8List bytes, int w, int h, int stride) {
  final r = min(kSize / h, kSize / w);
  final nw = (w * r).round(), nh = (h * r).round();
  final left = (kSize - nw) ~/ 2, top = (kSize - nh) ~/ 2;
  const plan = kSize * kSize;

  final out = Float32List(3 * plan)..fillRange(0, 3 * plan, 114 / 255.0);
  for (var y = 0; y < nh; y++) {
    final sy = min(h - 1, (y / r).floor());
    for (var x = 0; x < nw; x++) {
      final sx = min(w - 1, (x / r).floor());
      final i = sy * stride + sx * 4;
      final o = (top + y) * kSize + (left + x);
      out[o] = bytes[i + 2] / 255.0;
      out[plan + o] = bytes[i + 1] / 255.0;
      out[2 * plan + o] = bytes[i] / 255.0;
    }
  }
  return out;
}

Prediction predire(OrtSession session, Float32List data) {
  final input = OrtValueTensor.createTensorWithDataList(data, [1, 3, kSize, kSize]);
  final runOptions = OrtRunOptions();
  final sw = Stopwatch()..start();
  final outputs = session.run(runOptions, {'images': input});
  sw.stop();
  final ms = sw.elapsedMicroseconds / 1000;

  final out = (outputs[0]!.value as List)[0] as List;
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
        final entree = letterbox(m[0] as Uint8List, m[1] as int, m[2] as int, m[3] as int);
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
  String _tailleCamera = '';

  String? _lettre;
  double _score = 0;
  double _msInference = 0;
  double _msPreparation = 0;

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
      final raw = await rootBundle.load(kModele);
      final port = ReceivePort();
      _depuisTravailleur = port;
      port.listen(_surReponse);
      _isolate = await Isolate.spawn(
          _travailleur, <Object>[port.sendPort, raw.buffer.asUint8List()]);

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

  void _surImage(CameraImage img) {
    final versTravailleur = _versTravailleur;
    if (_occupe || versTravailleur == null) return;
    _occupe = true;
    final p = img.planes[0];
    _tailleCamera = '${img.width}x${img.height}';
    versTravailleur.send(<Object>[p.bytes, img.width, img.height, p.bytesPerRow]);
  }

  void _surReponse(dynamic msg) {
    if (msg is SendPort) {
      _versTravailleur = msg;
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
                          Text(_lettre ?? '-',
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
                        Text('${_fps.toStringAsFixed(0)} FPS · caméra $_tailleCamera',
                            style: const TextStyle(color: Colors.white70)),
                      ]),
                    ]),
                    const SizedBox(height: 8),
                    FilledButton(
                      onPressed: _mesure ? null : _lancerMesure,
                      child: Text(_mesure ? 'Mesure… $_restant s' : 'Mesurer 10 s'),
                    ),
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
                              child: const Text('Retour'))),
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