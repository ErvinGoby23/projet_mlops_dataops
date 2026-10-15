import 'dart:async';

import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:ultralytics_yolo/ultralytics_yolo.dart';

void main() => runApp(const AslApp());

class AslApp extends StatelessWidget {
  const AslApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'ASL Signes',
      debugShowCheckedModeBanner: false,
      theme: ThemeData(
        useMaterial3: true,
        brightness: Brightness.dark,
        colorSchemeSeed: const Color(0xFFFF6B35),
      ),
      home: const DetectionPage(),
    );
  }
}

/// Les versions CoreML embarquées (générées par GitHub Actions).
class ModelVariant {
  final String id;
  final String label;
  const ModelVariant(this.id, this.label);
  String get path => 'assets/models/asl_$id.mlpackage.zip';
}

const variants = [
  ModelVariant('int8_416', 'INT8 416'),
  ModelVariant('int8_320', 'INT8 320'),
  ModelVariant('fp16_416', 'FP16 416'),
  ModelVariant('int8_640', 'INT8 640'),
  ModelVariant('fp32_640', 'FP32 640'),
];

class DetectionPage extends StatefulWidget {
  const DetectionPage({super.key});

  @override
  State<DetectionPage> createState() => _DetectionPageState();
}

class _DetectionPageState extends State<DetectionPage> {
  final _controller = YOLOViewController();
  int _selected = 0;
  bool _switching = false;

  // Épellation : une lettre est validée si elle reste stable assez longtemps
  static const _minConfidence = 0.55;
  static const _holdDuration = Duration(milliseconds: 900);
  String? _letter;
  double _confidence = 0;
  String? _candidate;
  DateTime _candidateSince = DateTime.now();
  bool _alreadyAdded = false;
  double _progress = 0;
  String _word = '';

  // FPS instantané (moyenne glissante sur 1 s)
  int _frames = 0;
  double _fps = 0;
  Timer? _fpsTimer;

  // Mesure de 10 s pour le rapport
  static const _measureSeconds = 10;
  bool _measuring = false;
  int _measureFrames = 0;
  int _measureLeft = 0;
  Timer? _measureTimer;
  final Map<String, String> _measures = {};
  final List<double> _measureMs = [];   // temps d'inférence natif (Neural Engine) pendant la mesure
  double _lastMs = 0;
  double _nativeFps = 0;
  String? _modelError;

  @override
  void initState() {
    super.initState();
    _fpsTimer = Timer.periodic(const Duration(seconds: 1), (_) {
      if (!mounted) return;
      setState(() {
        _fps = _frames.toDouble();
        _frames = 0;
      });
    });
  }

  @override
  void dispose() {
    _fpsTimer?.cancel();
    _measureTimer?.cancel();
    super.dispose();
  }

  Future<void> _switchModel(int index) async {
    if (index == _selected || _switching || _measuring) return;
    setState(() {
      _switching = true;
      _selected = index;
      _modelError = null;
    });
    try {
      await _controller.switchModel(variants[index].path, YOLOTask.detect);
    } catch (e) {
      if (mounted) setState(() => _modelError = '$e');
    }
    if (mounted) setState(() => _switching = false);
  }

  void _startMeasure() {
    if (_measuring || _switching) return;
    setState(() {
      _measuring = true;
      _measureFrames = 0;
      _measureMs.clear();
      _measureLeft = _measureSeconds;
    });
    _measureTimer = Timer.periodic(const Duration(seconds: 1), (t) {
      if (!mounted) return;
      setState(() => _measureLeft--);
      if (_measureLeft <= 0) {
        t.cancel();
        final fps = _measureFrames / _measureSeconds;
        String ms = 'n/a';
        if (_measureMs.isNotEmpty) {
          final sorted = [..._measureMs]..sort();
          final mean = _measureMs.reduce((a, b) => a + b) / _measureMs.length;
          final median = sorted[sorted.length ~/ 2];
          ms = 'moy ${mean.toStringAsFixed(1)} ms | méd ${median.toStringAsFixed(1)} ms';
        }
        setState(() {
          _measures[variants[_selected].label] =
              '${fps.toStringAsFixed(1)} FPS | inférence $ms';
          _measuring = false;
        });
        HapticFeedback.heavyImpact();
      }
    });
  }

  void _onMetrics(YOLOPerformanceMetrics m) {
    if (_measuring && m.processingTimeMs > 0) _measureMs.add(m.processingTimeMs);
    if (!mounted) return;
    setState(() {
      _lastMs = m.processingTimeMs;
      _nativeFps = m.fps;
    });
  }

  String _resultsText() => [
        'Mesures iPhone (${_measureSeconds}s par modèle, stream caméra)',
        for (final e in _measures.entries) '${e.key} : ${e.value}',
      ].join('\n');

  void _onResult(List<dynamic> results) {
    _frames++;
    if (_measuring) _measureFrames++;

    dynamic best;
    for (final r in results) {
      if (best == null || r.confidence > best.confidence) best = r;
    }
    final String? letter = (best != null && best.confidence >= _minConfidence)
        ? (best.className as String).toUpperCase()
        : null;

    final now = DateTime.now();
    if (letter != _candidate) {
      _candidate = letter;
      _candidateSince = now;
      _alreadyAdded = false;
    }
    final held = now.difference(_candidateSince);
    double progress = 0;
    if (letter != null && !_alreadyAdded) {
      progress = (held.inMilliseconds / _holdDuration.inMilliseconds).clamp(0.0, 1.0);
      if (held >= _holdDuration) {
        _word += letter;
        _alreadyAdded = true;
        progress = 0;
        HapticFeedback.mediumImpact();
      }
    }

    if (!mounted) return;
    setState(() {
      _letter = letter;
      _confidence = best == null ? 0 : (best.confidence as num).toDouble();
      _progress = progress;
    });
  }

  Widget _panel({required Widget child}) => Container(
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
    final scheme = Theme.of(context).colorScheme;
    return Scaffold(
      backgroundColor: Colors.black,
      body: Stack(
        fit: StackFit.expand,
        children: [
          YOLOView(
            modelPath: variants[0].path,
            task: YOLOTask.detect,
            controller: _controller,
            streamingConfig: YOLOStreamingConfig.highPerformance(),
            onResult: _onResult,
            onPerformanceMetrics: _onMetrics,
            onModelError: (error, path, task) {
              if (mounted) setState(() => _modelError = '$error');
            },
          ),

          // ---- Haut : choix du modèle + lettre + FPS ----
          SafeArea(
            child: Column(
              children: [
                SingleChildScrollView(
                  scrollDirection: Axis.horizontal,
                  padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 6),
                  child: Row(
                    children: [
                      for (var i = 0; i < variants.length; i++)
                        Padding(
                          padding: const EdgeInsets.only(right: 6),
                          child: ChoiceChip(
                            label: Text(variants[i].label),
                            selected: i == _selected,
                            onSelected: (_) => _switchModel(i),
                          ),
                        ),
                    ],
                  ),
                ),
                _panel(
                  child: Row(
                    mainAxisSize: MainAxisSize.min,
                    children: [
                      SizedBox(
                        width: 56,
                        height: 56,
                        child: Stack(alignment: Alignment.center, children: [
                          CircularProgressIndicator(
                            value: _progress,
                            strokeWidth: 4,
                            color: scheme.primary,
                            backgroundColor: Colors.white12,
                          ),
                          Text(_letter ?? '–',
                              style: const TextStyle(fontSize: 30, fontWeight: FontWeight.bold)),
                        ]),
                      ),
                      const SizedBox(width: 14),
                      Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
                        Text(_switching
                            ? 'Chargement ${variants[_selected].label}…'
                            : 'Modèle : ${variants[_selected].label}'),
                        Text('Confiance : ${(_confidence * 100).toStringAsFixed(0)} %'),
                        Text('${_fps.toStringAsFixed(0)} FPS app · ${_nativeFps.toStringAsFixed(0)} FPS natif',
                            style: const TextStyle(color: Colors.white70)),
                        Text('Inférence : ${_lastMs.toStringAsFixed(1)} ms',
                            style: const TextStyle(color: Colors.white70)),
                        if (_modelError != null)
                          SizedBox(
                            width: 230,
                            child: Text('❌ $_modelError',
                                maxLines: 3,
                                overflow: TextOverflow.ellipsis,
                                style: const TextStyle(color: Colors.redAccent, fontSize: 12)),
                          ),
                      ]),
                    ],
                  ),
                ),
              ],
            ),
          ),

          // ---- Bas : mesures + mot épelé ----
          SafeArea(
            child: Align(
              alignment: Alignment.bottomCenter,
              child: Column(
                mainAxisSize: MainAxisSize.min,
                children: [
                  _panel(
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.stretch,
                      children: [
                        FilledButton.icon(
                          onPressed: _measuring || _switching ? null : _startMeasure,
                          icon: const Icon(Icons.timer_outlined),
                          label: Text(_measuring
                              ? 'Mesure en cours… $_measureLeft s'
                              : 'Mesurer 10 s (${variants[_selected].label})'),
                        ),
                        if (_measures.isNotEmpty) ...[
                          const SizedBox(height: 8),
                          for (final e in _measures.entries)
                            Text('${e.key} : ${e.value}',
                                style: const TextStyle(fontFamily: 'Menlo', fontSize: 12)),
                          const SizedBox(height: 6),
                          OutlinedButton.icon(
                            onPressed: () {
                              Clipboard.setData(ClipboardData(text: _resultsText()));
                              ScaffoldMessenger.of(context).showSnackBar(
                                  const SnackBar(content: Text('Résultats copiés')));
                            },
                            icon: const Icon(Icons.copy, size: 16),
                            label: const Text('Copier les résultats'),
                          ),
                        ],
                      ],
                    ),
                  ),
                  _panel(
                    child: Column(
                      mainAxisSize: MainAxisSize.min,
                      children: [
                        Text(
                          _word.isEmpty ? 'Tiens une lettre ~1 s pour l\'écrire' : _word,
                          textAlign: TextAlign.center,
                          style: TextStyle(
                            fontSize: _word.isEmpty ? 15 : 32,
                            fontWeight: FontWeight.w600,
                            letterSpacing: 2,
                            color: _word.isEmpty ? Colors.white60 : Colors.white,
                          ),
                        ),
                        const SizedBox(height: 10),
                        Row(children: [
                          Expanded(
                              child: FilledButton.tonal(
                                  onPressed: () => setState(() => _word += ' '),
                                  child: const Text('Espace'))),
                          const SizedBox(width: 8),
                          Expanded(
                              child: FilledButton.tonal(
                                  onPressed: _word.isEmpty
                                      ? null
                                      : () => setState(
                                          () => _word = _word.substring(0, _word.length - 1)),
                                  child: const Text('⌫'))),
                          const SizedBox(width: 8),
                          Expanded(
                              child: FilledButton(
                                  onPressed: () => setState(() => _word = ''),
                                  child: const Text('Effacer'))),
                        ]),
                      ],
                    ),
                  ),
                ],
              ),
            ),
          ),
        ],
      ),
    );
  }
}
