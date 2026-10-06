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

class DetectionPage extends StatefulWidget {
  const DetectionPage({super.key});

  @override
  State<DetectionPage> createState() => _DetectionPageState();
}

class _DetectionPageState extends State<DetectionPage> {
  // Modèle embarqué dans l'app (CoreML quantifié, généré par GitHub Actions)
  static const _modelPath = 'assets/models/asl.mlpackage.zip';

  // Épellation : une lettre est validée si elle reste stable assez longtemps
  static const _minConfidence = 0.55;
  static const _holdDuration = Duration(milliseconds: 900);

  String? _letter; // lettre vue en ce moment
  double _confidence = 0;
  String? _candidate; // lettre en cours de validation
  DateTime _candidateSince = DateTime.now();
  bool _alreadyAdded = false; // évite d'ajouter 10 fois la même lettre
  double _progress = 0; // 0 → 1 pendant le maintien
  String _word = '';

  int _frames = 0;
  double _fps = 0;
  Timer? _fpsTimer;

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
    super.dispose();
  }

  void _onResult(List<dynamic> results) {
    _frames++;

    // Meilleure détection de l'image
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

  @override
  Widget build(BuildContext context) {
    final scheme = Theme.of(context).colorScheme;
    return Scaffold(
      backgroundColor: Colors.black,
      body: Stack(
        fit: StackFit.expand,
        children: [
          // Caméra + inférence YOLO sur l'iPhone (Neural Engine via CoreML)
          YOLOView(
            modelPath: _modelPath,
            task: YOLOTask.detect,
            onResult: _onResult,
          ),

          // ---- Bandeau du haut : lettre + stats ----
          SafeArea(
            child: Align(
              alignment: Alignment.topCenter,
              child: Container(
                margin: const EdgeInsets.all(12),
                padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 10),
                decoration: BoxDecoration(
                  color: Colors.black.withValues(alpha: 0.6),
                  borderRadius: BorderRadius.circular(16),
                ),
                child: Row(
                  mainAxisSize: MainAxisSize.min,
                  children: [
                    SizedBox(
                      width: 56,
                      height: 56,
                      child: Stack(
                        alignment: Alignment.center,
                        children: [
                          CircularProgressIndicator(
                            value: _progress,
                            strokeWidth: 4,
                            color: scheme.primary,
                            backgroundColor: Colors.white12,
                          ),
                          Text(
                            _letter ?? '–',
                            style: const TextStyle(
                                fontSize: 30, fontWeight: FontWeight.bold),
                          ),
                        ],
                      ),
                    ),
                    const SizedBox(width: 14),
                    Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        Text('Confiance : ${(_confidence * 100).toStringAsFixed(0)} %'),
                        Text('${_fps.toStringAsFixed(0)} FPS · sur l\'iPhone',
                            style: const TextStyle(color: Colors.white70)),
                      ],
                    ),
                  ],
                ),
              ),
            ),
          ),

          // ---- Bas : mot épelé + boutons ----
          SafeArea(
            child: Align(
              alignment: Alignment.bottomCenter,
              child: Container(
                margin: const EdgeInsets.all(12),
                padding: const EdgeInsets.all(14),
                decoration: BoxDecoration(
                  color: Colors.black.withValues(alpha: 0.7),
                  borderRadius: BorderRadius.circular(16),
                ),
                child: Column(
                  mainAxisSize: MainAxisSize.min,
                  children: [
                    Text(
                      _word.isEmpty ? 'Tiens une lettre ~1 s pour l\'écrire' : _word,
                      textAlign: TextAlign.center,
                      style: TextStyle(
                        fontSize: _word.isEmpty ? 16 : 34,
                        fontWeight: FontWeight.w600,
                        letterSpacing: 2,
                        color: _word.isEmpty ? Colors.white60 : Colors.white,
                      ),
                    ),
                    const SizedBox(height: 12),
                    Row(
                      children: [
                        Expanded(
                          child: FilledButton.tonal(
                            onPressed: () => setState(() => _word += ' '),
                            child: const Text('Espace'),
                          ),
                        ),
                        const SizedBox(width: 8),
                        Expanded(
                          child: FilledButton.tonal(
                            onPressed: _word.isEmpty
                                ? null
                                : () => setState(() =>
                                    _word = _word.substring(0, _word.length - 1)),
                            child: const Text('⌫'),
                          ),
                        ),
                        const SizedBox(width: 8),
                        Expanded(
                          child: FilledButton(
                            onPressed: () => setState(() => _word = ''),
                            child: const Text('Effacer'),
                          ),
                        ),
                      ],
                    ),
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
