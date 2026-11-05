import 'package:flutter/material.dart';

import 'onnx_page.dart';

void main() => runApp(const AslApp());

class AslApp extends StatelessWidget {
  const AslApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'ASL ONNX',
      debugShowCheckedModeBanner: false,
      theme: ThemeData(
        useMaterial3: true,
        brightness: Brightness.dark,
        colorSchemeSeed: const Color(0xFFFF6B35),
      ),
      home: const OnnxPage(),
    );
  }
}
