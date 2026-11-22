import 'dart:convert';
import 'dart:typed_data';
import 'package:file_picker/file_picker.dart';
import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;

void main() {
  runApp(const LungPathologyApp());
}

class LungPathologyApp extends StatelessWidget {
  const LungPathologyApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'AI Chest Disease Detection - RACM',
      debugShowCheckedModeBanner: false,
      theme: ThemeData.dark().copyWith(
        scaffoldBackgroundColor: const Color(0xFF070B14),
        primaryColor: const Color(0xFF00F0FF),
        colorScheme: const ColorScheme.dark(
          primary: Color(0xFF00F0FF),
          secondary: Color(0xFFA855F7),
          surface: Color(0xFF0D1527),
        ),
      ),
      home: const DashboardScreen(),
    );
  }
}

class DashboardScreen extends StatefulWidget {
  const DashboardScreen({super.key});

  @override
  State<DashboardScreen> createState() => _DashboardScreenState();
}

class _DashboardScreenState extends State<DashboardScreen> {
  // Configuración de Grad-CAM
  String _gradcamMethod = 'gradcam++';
  double _alphaOverlay = 0.42;
  String _cmapChoice = 'jet';
  String _apiUrl = 'http://127.0.0.1:8000';

  // Estado de análisis e imagen
  Uint8List? _originalImageBytes;
  String? _imageFileName;
  Uint8List? _heatmapOverlayBytes;
  Map<String, dynamic>? _diagnosticResult;
  bool _isAnalyzing = false;
  String? _errorMessage;

  // Servidor backend status
  bool _backendConnected = false;

  @override
  void initState() {
    super.initState();
    _checkBackendHealth();
  }

  Future<void> _checkBackendHealth() async {
    try {
      final response = await http
          .get(Uri.parse('$_apiUrl/health'))
          .timeout(const Duration(seconds: 2));
      if (response.statusCode == 200) {
        if (!mounted) return;
        setState(() {
          _backendConnected = true;
        });
      } else {
        if (!mounted) return;
        setState(() {
          _backendConnected = false;
        });
      }
    } catch (_) {
      if (!mounted) return;
      setState(() {
        _backendConnected = false;
      });
    }
  }

  void _showEditApiDialog() {
    final controller = TextEditingController(text: _apiUrl);
    showDialog(
      context: context,
      builder: (ctx) => AlertDialog(
        backgroundColor: const Color(0xFF0F172A),
        title: const Text('Configurar Servidor de Inferencia', style: TextStyle(color: Colors.white, fontSize: 16)),
        content: Column(
          mainAxisSize: MainAxisSize.min,
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            const Text(
              'Ingresa la IP o URL del servidor Python:',
              style: TextStyle(color: Color(0xFF94A3B8), fontSize: 13),
            ),
            const SizedBox(height: 8),
            TextField(
              controller: controller,
              style: const TextStyle(color: Colors.white, fontSize: 13),
              decoration: const InputDecoration(
                hintText: 'http://192.168.100.15:8000',
                hintStyle: TextStyle(color: Color(0xFF475569)),
                enabledBorder: UnderlineInputBorder(borderSide: BorderSide(color: Color(0xFF00F0FF))),
              ),
            ),
          ],
        ),
        actions: [
          TextButton(
            onPressed: () => Navigator.pop(ctx),
            child: const Text('Cancelar', style: TextStyle(color: Color(0xFF94A3B8))),
          ),
          ElevatedButton(
            style: ElevatedButton.styleFrom(backgroundColor: const Color(0xFF00F0FF)),
            onPressed: () {
              final newUrl = controller.text.trim();
              if (newUrl.isNotEmpty) {
                setState(() => _apiUrl = newUrl);
                Navigator.pop(ctx);
                _checkBackendHealth();
              }
            },
            child: const Text('Guardar y Probar', style: TextStyle(color: Colors.black, fontWeight: FontWeight.bold)),
          ),
        ],
      ),
    );
  }

  Future<void> _pickImage() async {
    try {
      final result = await FilePicker.platform.pickFiles(
        type: FileType.custom,
        allowedExtensions: ['jpg', 'jpeg', 'png'],
        withData: true,
      );

      if (result != null && result.files.isNotEmpty) {
        final file = result.files.first;
        if (file.bytes != null) {
          setState(() {
            _originalImageBytes = file.bytes;
            _imageFileName = file.name;
            _heatmapOverlayBytes = null;
            _diagnosticResult = null;
            _errorMessage = null;
          });
          await _executeInference(file.bytes!, file.name);
        }
      }
    } catch (e) {
      if (!mounted) return;
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text('Error al seleccionar imagen: $e')),
      );
    }
  }

  Future<void> _executeInference(Uint8List imageBytes, String fileName) async {
    setState(() {
      _isAnalyzing = true;
      _errorMessage = null;
    });

    try {
      final base64Image = base64Encode(imageBytes);
      final payload = jsonEncode({
        'image_b64': base64Image,
        'method': _gradcamMethod,
        'alpha': _alphaOverlay,
        'cmap': _cmapChoice,
        'class_index': -1,
      });

      final response = await http
          .post(
            Uri.parse('$_apiUrl/predict'),
            headers: {'Content-Type': 'application/json'},
            body: payload,
          )
          .timeout(const Duration(seconds: 45));

      if (response.statusCode == 200) {
        final data = jsonDecode(response.body) as Map<String, dynamic>;
        final overlayB64 = data['overlay_b64'] as String?;

        Uint8List? overlayBytes;
        if (overlayB64 != null) {
          overlayBytes = base64Decode(overlayB64);
        }

        final probsRaw = data['probabilidades'] as Map<String, dynamic>? ?? {};
        final Map<String, double> probs = {};
        probsRaw.forEach((k, v) {
          probs[k] = (v as num).toDouble();
        });

        setState(() {
          _backendConnected = true;
          _heatmapOverlayBytes = overlayBytes;
          _diagnosticResult = {
            'predicted_class': data['clase_detectada'] ?? 'Desconocido',
            'explained_class': data['clase_explicada'] ?? '',
            'confidence': (data['confianza'] as num?)?.toDouble() ?? 0.0,
            'is_healthy': data['es_sano'] ?? false,
            'method': data['metodo_gradcam'] ?? _gradcamMethod,
            'probabilities': probs,
          };
          _isAnalyzing = false;
        });
      } else {
        final err = jsonDecode(response.body);
        setState(() {
          _errorMessage = err['error'] ?? 'Error devuelto por el servidor (${response.statusCode})';
          _isAnalyzing = false;
        });
      }
    } catch (e) {
      setState(() {
        _backendConnected = false;
        _errorMessage =
            'No se pudo conectar al motor de inferencia en $_apiUrl.\n'
            'Inicia el servidor Python ejecutando en la terminal:\n'
            'python src/api.py';
        _isAnalyzing = false;
      });
    }
  }

  @override
  Widget build(BuildContext context) {
    final screenWidth = MediaQuery.of(context).size.width;
    final isDesktop = screenWidth > 960;

    return Scaffold(
      appBar: AppBar(
        backgroundColor: const Color(0xFF090E1A),
        elevation: 0,
        title: Row(
          children: [
            Container(
              width: 40,
              height: 40,
              decoration: BoxDecoration(
                shape: BoxShape.circle,
                gradient: const LinearGradient(
                  colors: [Color(0xFF00F0FF), Color(0xFFA855F7)],
                ),
                boxShadow: [
                  BoxShadow(
                    color: const Color(0xFF00F0FF).withValues(alpha: 0.35),
                    blurRadius: 10,
                  ),
                ],
              ),
              child: ClipOval(
                child: Image.asset(
                  'assets/racm_logo.jpg',
                  fit: BoxFit.cover,
                  errorBuilder: (ctx, err, stack) => const Icon(
                    Icons.medical_services_outlined,
                    color: Colors.white,
                    size: 20,
                  ),
                ),
              ),
            ),
            const SizedBox(width: 14),
            Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                ShaderMask(
                  shaderCallback: (bounds) => const LinearGradient(
                    colors: [Color(0xFF00F0FF), Color(0xFFA855F7)],
                  ).createShader(bounds),
                  child: const Text(
                    'AI Chest Disease Detection',
                    style: TextStyle(
                      fontSize: 16,
                      fontWeight: FontWeight.bold,
                      color: Colors.white,
                    ),
                  ),
                ),
                const Text(
                  'Deep Learning & Grad-CAM++ Interpretability',
                  style: TextStyle(fontSize: 11, color: Color(0xFF94A3B8)),
                ),
              ],
            ),
          ],
        ),
        actions: [
          IconButton(
            tooltip: 'Revisar conexión con el motor de inferencia',
            icon: Icon(
              _backendConnected ? Icons.cloud_done : Icons.cloud_off,
              color: _backendConnected ? const Color(0xFF10B981) : const Color(0xFFF43F5E),
            ),
            onPressed: _checkBackendHealth,
          ),
          const SizedBox(width: 16),
        ],
      ),
      drawer: !isDesktop ? Drawer(child: _buildSidebarContent()) : null,
      body: Row(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          if (isDesktop)
            Container(
              width: 320,
              decoration: const BoxDecoration(
                color: Color(0xFF090E1A),
                border: Border(
                  right: BorderSide(color: Color(0x2600F0FF), width: 1),
                ),
              ),
              child: _buildSidebarContent(),
            ),
          Expanded(
            child: SingleChildScrollView(
              padding: const EdgeInsets.symmetric(horizontal: 28, vertical: 24),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.stretch,
                children: [
                  _buildHeroBanner(),
                  const SizedBox(height: 24),
                  _buildUploadSection(),
                  const SizedBox(height: 24),
                  if (_errorMessage != null) _buildErrorBanner(),
                  if (_isAnalyzing) _buildLoadingCard(),
                  if (!_isAnalyzing && _originalImageBytes == null)
                    _buildPlaceholderCard(),
                  if (!_isAnalyzing && _originalImageBytes != null && _diagnosticResult != null)
                    _buildResultsGrid(isDesktop),
                ],
              ),
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildSidebarContent() {
    return ListView(
      padding: const EdgeInsets.all(20),
      children: [
        _buildCreatorCard(),
        const SizedBox(height: 24),
        const Text(
          '⚙️ Configuración Visual',
          style: TextStyle(
            fontSize: 15,
            fontWeight: FontWeight.bold,
            color: Color(0xFFE2E8F0),
          ),
        ),
        const SizedBox(height: 16),
        const Text(
          'Método de explicabilidad',
          style: TextStyle(fontSize: 12, color: Color(0xFF94A3B8)),
        ),
        const SizedBox(height: 6),
        Container(
          padding: const EdgeInsets.symmetric(horizontal: 12),
          decoration: BoxDecoration(
            color: const Color(0xFF0F172A),
            borderRadius: BorderRadius.circular(10),
            border: Border.all(color: const Color(0x3300F0FF)),
          ),
          child: DropdownButtonHideUnderline(
            child: DropdownButton<String>(
              value: _gradcamMethod,
              isExpanded: true,
              dropdownColor: const Color(0xFF0F172A),
              style: const TextStyle(color: Colors.white, fontSize: 13),
              items: const [
                DropdownMenuItem(
                  value: 'gradcam++',
                  child: Text('Grad-CAM++ (Alta precisión focal)'),
                ),
                DropdownMenuItem(
                  value: 'gradcam',
                  child: Text('Grad-CAM clásico'),
                ),
              ],
              onChanged: (val) {
                if (val != null) {
                  setState(() => _gradcamMethod = val);
                  if (_originalImageBytes != null) {
                    _executeInference(_originalImageBytes!, _imageFileName ?? 'image.jpg');
                  }
                }
              },
            ),
          ),
        ),
        const SizedBox(height: 18),
        Row(
          mainAxisAlignment: MainAxisAlignment.spaceBetween,
          children: [
            const Text(
              'Opacidad del mapa',
              style: TextStyle(fontSize: 12, color: Color(0xFF94A3B8)),
            ),
            Text(
              _alphaOverlay.toStringAsFixed(2),
              style: const TextStyle(fontSize: 12, color: Color(0xFF00F0FF), fontWeight: FontWeight.bold),
            ),
          ],
        ),
        SliderTheme(
          data: SliderTheme.of(context).copyWith(
            activeTrackColor: const Color(0xFF00F0FF),
            inactiveTrackColor: const Color(0xFF1E293B),
            thumbColor: const Color(0xFFA855F7),
          ),
          child: Slider(
            value: _alphaOverlay,
            min: 0.15,
            max: 0.75,
            divisions: 12,
            onChanged: (val) {
              setState(() => _alphaOverlay = val);
            },
            onChangeEnd: (val) {
              if (_originalImageBytes != null) {
                _executeInference(_originalImageBytes!, _imageFileName ?? 'image.jpg');
              }
            },
          ),
        ),
        const SizedBox(height: 12),
        const Text(
          'Paleta térmica (Colormap)',
          style: TextStyle(fontSize: 12, color: Color(0xFF94A3B8)),
        ),
        const SizedBox(height: 6),
        Container(
          padding: const EdgeInsets.symmetric(horizontal: 12),
          decoration: BoxDecoration(
            color: const Color(0xFF0F172A),
            borderRadius: BorderRadius.circular(10),
            border: Border.all(color: const Color(0x3300F0FF)),
          ),
          child: DropdownButtonHideUnderline(
            child: DropdownButton<String>(
              value: _cmapChoice,
              isExpanded: true,
              dropdownColor: const Color(0xFF0F172A),
              style: const TextStyle(color: Colors.white, fontSize: 13),
              items: const [
                DropdownMenuItem(value: 'jet', child: Text('jet (Estándar clínico)')),
                DropdownMenuItem(value: 'inferno', child: Text('inferno (Alto contraste)')),
                DropdownMenuItem(value: 'turbo', child: Text('turbo (Perceptual suave)')),
              ],
              onChanged: (val) {
                if (val != null) {
                  setState(() => _cmapChoice = val);
                  if (_originalImageBytes != null) {
                    _executeInference(_originalImageBytes!, _imageFileName ?? 'image.jpg');
                  }
                }
              },
            ),
          ),
        ),
        const SizedBox(height: 24),
        Container(
          padding: const EdgeInsets.all(12),
          decoration: BoxDecoration(
            color: const Color(0xFF0D1527),
            borderRadius: BorderRadius.circular(10),
            border: Border.all(
              color: _backendConnected ? const Color(0x4010B981) : const Color(0x40F43F5E),
            ),
          ),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Row(
                children: [
                  Icon(
                    _backendConnected ? Icons.check_circle : Icons.warning_amber_rounded,
                    size: 16,
                    color: _backendConnected ? const Color(0xFF10B981) : const Color(0xFFF43F5E),
                  ),
                  const SizedBox(width: 8),
                  Text(
                    _backendConnected ? 'Motor conectado' : 'Motor desconectado',
                    style: TextStyle(
                      fontSize: 12,
                      fontWeight: FontWeight.bold,
                      color: _backendConnected ? const Color(0xFF10B981) : const Color(0xFFF43F5E),
                    ),
                  ),
                ],
              ),
              const SizedBox(height: 4),
              Row(
                mainAxisAlignment: MainAxisAlignment.spaceBetween,
                children: [
                  Expanded(
                    child: Text(
                      'Endpoint: $_apiUrl',
                      style: const TextStyle(fontSize: 11, color: Color(0xFF64748B)),
                      overflow: TextOverflow.ellipsis,
                    ),
                  ),
                  InkWell(
                    onTap: _showEditApiDialog,
                    child: const Padding(
                      padding: EdgeInsets.symmetric(horizontal: 4, vertical: 2),
                      child: Icon(Icons.edit, size: 14, color: Color(0xFF00F0FF)),
                    ),
                  ),
                ],
              ),
            ],
          ),
        ),
      ],
    );
  }

  Widget _buildCreatorCard() {
    return Container(
      padding: const EdgeInsets.all(18),
      decoration: BoxDecoration(
        gradient: const LinearGradient(
          begin: Alignment.topLeft,
          end: Alignment.bottomRight,
          colors: [Color(0xE610182C), Color(0xA61A103C)],
        ),
        borderRadius: BorderRadius.circular(18),
        border: Border.all(color: const Color(0x4D00F0FF)),
        boxShadow: [
          BoxShadow(
            color: const Color(0xFF00F0FF).withValues(alpha: 0.12),
            blurRadius: 18,
            offset: const Offset(0, 6),
          ),
        ],
      ),
      child: Column(
        children: [
          Container(
            width: 90,
            height: 90,
            padding: const EdgeInsets.all(3),
            decoration: const BoxDecoration(
              shape: BoxShape.circle,
              gradient: LinearGradient(
                colors: [Color(0xFF00F0FF), Color(0xFFA855F7)],
              ),
            ),
            child: ClipOval(
              child: Image.asset(
                'assets/racm_logo.jpg',
                fit: BoxFit.cover,
                errorBuilder: (ctx, err, stack) => Container(
                  color: const Color(0xFF0B0F19),
                  child: const Center(
                    child: Text('RACM', style: TextStyle(fontWeight: FontWeight.bold, color: Color(0xFF00F0FF))),
                  ),
                ),
              ),
            ),
          ),
          const SizedBox(height: 12),
          const Text(
            'Roberto Alejandro\nChagra Martínez',
            textAlign: TextAlign.center,
            style: TextStyle(
              fontSize: 15,
              fontWeight: FontWeight.bold,
              color: Colors.white,
              height: 1.25,
            ),
          ),
          const SizedBox(height: 8),
          Container(
            padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 4),
            decoration: BoxDecoration(
              color: const Color(0x3300F0FF),
              borderRadius: BorderRadius.circular(20),
              border: Border.all(color: const Color(0x6600F0FF)),
            ),
            child: const Row(
              mainAxisSize: MainAxisSize.min,
              children: [
                Icon(Icons.circle, color: Color(0xFF00F0FF), size: 7),
                SizedBox(width: 6),
                Text(
                  'Creador del Modelo',
                  style: TextStyle(
                    fontSize: 11,
                    fontWeight: FontWeight.w600,
                    color: Color(0xFF00F0FF),
                  ),
                ),
              ],
            ),
          ),
          const SizedBox(height: 6),
          const Text(
            'Artificial Intelligence & Data Science',
            textAlign: TextAlign.center,
            style: TextStyle(fontSize: 11, color: Color(0xFF94A3B8)),
          ),
          const SizedBox(height: 10),
          const Wrap(
            spacing: 6,
            runSpacing: 4,
            alignment: WrapAlignment.center,
            children: [
              _TagBadge(label: 'RACM AI'),
              _TagBadge(label: 'Deep Learning'),
              _TagBadge(label: 'Computer Vision'),
            ],
          ),
        ],
      ),
    );
  }

  Widget _buildHeroBanner() {
    return Container(
      padding: const EdgeInsets.all(24),
      decoration: BoxDecoration(
        gradient: const LinearGradient(
          begin: Alignment.topLeft,
          end: Alignment.bottomRight,
          colors: [Color(0xFF0D1527), Color(0xFF090E1A)],
        ),
        borderRadius: BorderRadius.circular(20),
        border: Border.all(color: const Color(0x3300F0FF)),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Container(
            padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 5),
            decoration: BoxDecoration(
              color: const Color(0x2600F0FF),
              borderRadius: BorderRadius.circular(20),
              border: Border.all(color: const Color(0x6600F0FF)),
            ),
            child: const Row(
              mainAxisSize: MainAxisSize.min,
              children: [
                Icon(Icons.circle, color: Color(0xFF00F0FF), size: 8),
                SizedBox(width: 8),
                Text(
                  'Sistema Inteligente de Diagnóstico Torácico',
                  style: TextStyle(
                    fontSize: 11,
                    fontWeight: FontWeight.bold,
                    color: Color(0xFF00F0FF),
                  ),
                ),
              ],
            ),
          ),
          const SizedBox(height: 12),
          const Text(
            'Detección de Enfermedades Pulmonares con IA',
            style: TextStyle(
              fontSize: 24,
              fontWeight: FontWeight.bold,
              color: Colors.white,
            ),
          ),
          const SizedBox(height: 8),
          const Text(
            'Análisis predictivo multiclase sobre radiografías de tórax (CXR) con interpretabilidad biológica en tiempo real mediante mapas de atención Grad-CAM / Grad-CAM++.',
            style: TextStyle(fontSize: 13, color: Color(0xFF94A3B8), height: 1.4),
          ),
          const SizedBox(height: 14),
          const Wrap(
            spacing: 8,
            runSpacing: 6,
            children: [
              Text('👤 Creador: Roberto Alejandro Chagra Martínez', style: TextStyle(fontSize: 12, color: Color(0xFFCBD5E1))),
              Text('•', style: TextStyle(color: Color(0xFF64748B))),
              Text('🔬 Disciplina: Artificial Intelligence & Data Science', style: TextStyle(fontSize: 12, color: Color(0xFFCBD5E1))),
              Text('•', style: TextStyle(color: Color(0xFF64748B))),
              Text('🏷️ RACM', style: TextStyle(fontSize: 12, color: Color(0xFF00F0FF), fontWeight: FontWeight.bold)),
            ],
          ),
        ],
      ),
    );
  }

  Widget _buildUploadSection() {
    return InkWell(
      onTap: _pickImage,
      borderRadius: BorderRadius.circular(16),
      child: Container(
        padding: const EdgeInsets.symmetric(horizontal: 24, vertical: 32),
        decoration: BoxDecoration(
          color: const Color(0xFF0D1527),
          borderRadius: BorderRadius.circular(16),
          border: Border.all(color: const Color(0x4D00F0FF), style: BorderStyle.solid),
        ),
        child: Column(
          children: [
            Container(
              width: 56,
              height: 56,
              decoration: BoxDecoration(
                shape: BoxShape.circle,
                gradient: const LinearGradient(
                  colors: [Color(0xFF00F0FF), Color(0xFFA855F7)],
                ),
                boxShadow: [
                  BoxShadow(
                    color: const Color(0xFF00F0FF).withValues(alpha: 0.3),
                    blurRadius: 14,
                  ),
                ],
              ),
              child: const Icon(Icons.cloud_upload_outlined, color: Colors.white, size: 28),
            ),
            const SizedBox(height: 14),
            Text(
              _imageFileName != null
                  ? 'Archivo seleccionado: $_imageFileName'
                  : '📂 Selecciona o arrastra una radiografía de tórax (JPG, JPEG, PNG)...',
              textAlign: TextAlign.center,
              style: const TextStyle(
                fontSize: 15,
                fontWeight: FontWeight.bold,
                color: Colors.white,
              ),
            ),
            const SizedBox(height: 6),
            const Text(
              'Haz clic aquí para examinar archivos de tu sistema',
              style: TextStyle(fontSize: 12, color: Color(0xFF64748B)),
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildErrorBanner() {
    return Container(
      margin: const EdgeInsets.only(bottom: 20),
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: const Color(0x33F43F5E),
        borderRadius: BorderRadius.circular(14),
        border: Border.all(color: const Color(0xFFF43F5E)),
      ),
      child: Row(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          const Icon(Icons.error_outline, color: Color(0xFFF43F5E), size: 24),
          const SizedBox(width: 14),
          Expanded(
            child: Text(
              _errorMessage!,
              style: const TextStyle(color: Color(0xFFFFE4E6), fontSize: 13, height: 1.4),
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildLoadingCard() {
    return Container(
      padding: const EdgeInsets.symmetric(vertical: 48),
      child: const Center(
        child: Column(
          children: [
            CircularProgressIndicator(color: Color(0xFF00F0FF)),
            SizedBox(height: 20),
            Text(
              '⚡ Ejecutando inferencia neuronal y calculando gradientes Grad-CAM++...',
              style: TextStyle(color: Color(0xFF00F0FF), fontSize: 14, fontWeight: FontWeight.bold),
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildPlaceholderCard() {
    return Container(
      padding: const EdgeInsets.symmetric(vertical: 48, horizontal: 24),
      decoration: BoxDecoration(
        color: const Color(0xFF0B101E),
        borderRadius: BorderRadius.circular(16),
        border: Border.all(color: const Color(0x1A00F0FF)),
      ),
      child: Column(
        children: [
          Icon(Icons.medical_services_outlined, size: 48, color: Colors.white.withValues(alpha: 0.2)),
          const SizedBox(height: 16),
          const Text(
            'Listo para el análisis radiológico',
            style: TextStyle(fontSize: 18, fontWeight: FontWeight.bold, color: Colors.white),
          ),
          const SizedBox(height: 8),
          const Text(
            'Sube una placa de tórax en el recuadro superior para que la red neuronal efectúe la clasificación multiclase y despliegue el mapa térmico de atención Grad-CAM++.',
            textAlign: TextAlign.center,
            style: TextStyle(fontSize: 13, color: Color(0xFF64748B)),
          ),
        ],
      ),
    );
  }

  Widget _buildResultsGrid(bool isDesktop) {
    final originalCol = Column(
      crossAxisAlignment: CrossAxisAlignment.stretch,
      children: [
        const Text(
          '🖼️ Radiografía Original',
          style: TextStyle(fontSize: 14, fontWeight: FontWeight.bold, color: Colors.white),
        ),
        const SizedBox(height: 8),
        ClipRRect(
          borderRadius: BorderRadius.circular(14),
          child: Image.memory(
            _originalImageBytes!,
            fit: BoxFit.contain,
            height: 260,
          ),
        ),
        const SizedBox(height: 4),
        const Text(
          'Entrada clínica (CXR)',
          textAlign: TextAlign.center,
          style: TextStyle(fontSize: 11, color: Color(0xFF64748B)),
        ),
      ],
    );

    final heatmapCol = Column(
      crossAxisAlignment: CrossAxisAlignment.stretch,
      children: [
        const Text(
          '🎯 Mapa de Atención Grad-CAM',
          style: TextStyle(fontSize: 14, fontWeight: FontWeight.bold, color: Colors.white),
        ),
        const SizedBox(height: 8),
        ClipRRect(
          borderRadius: BorderRadius.circular(14),
          child: _heatmapOverlayBytes != null
              ? Image.memory(
                  _heatmapOverlayBytes!,
                  fit: BoxFit.contain,
                  height: 260,
                )
              : Container(
                  height: 260,
                  decoration: BoxDecoration(
                    color: const Color(0xFF0F172A),
                    borderRadius: BorderRadius.circular(14),
                  ),
                  child: const Center(
                    child: Text('Generando mapa...', style: TextStyle(color: Color(0xFF64748B))),
                  ),
                ),
        ),
        const SizedBox(height: 4),
        Text(
          'Foco de atención (${(_diagnosticResult?['method'] ?? 'GRADCAM++').toString().toUpperCase()}) para: ${_diagnosticResult?['explained_class'] ?? ''}',
          textAlign: TextAlign.center,
          style: const TextStyle(fontSize: 11, color: Color(0xFF64748B)),
        ),
      ],
    );

    final diagnosticCol = _buildDiagnosticCard();

    if (isDesktop) {
      return Row(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Expanded(flex: 10, child: originalCol),
          const SizedBox(width: 18),
          Expanded(flex: 10, child: heatmapCol),
          const SizedBox(width: 18),
          Expanded(flex: 12, child: diagnosticCol),
        ],
      );
    } else {
      return Column(
        children: [
          originalCol,
          const SizedBox(height: 20),
          heatmapCol,
          const SizedBox(height: 20),
          diagnosticCol,
        ],
      );
    }
  }

  Widget _buildDiagnosticCard() {
    final isHealthy = _diagnosticResult?['is_healthy'] == true;
    final predictedClass = _diagnosticResult?['predicted_class'] ?? 'Desconocido';
    final confidence = (_diagnosticResult?['confidence'] as num?)?.toDouble() ?? 0.0;
    final probs = _diagnosticResult?['probabilities'] as Map<String, double>? ?? {};

    final accentColor = isHealthy ? const Color(0xFF10B981) : const Color(0xFFF43F5E);

    return Column(
      crossAxisAlignment: CrossAxisAlignment.stretch,
      children: [
        const Text(
          '📊 Diagnóstico del Modelo',
          style: TextStyle(fontSize: 14, fontWeight: FontWeight.bold, color: Colors.white),
        ),
        const SizedBox(height: 8),
        Container(
          padding: const EdgeInsets.all(18),
          decoration: BoxDecoration(
            gradient: LinearGradient(
              colors: [
                accentColor.withValues(alpha: 0.18),
                const Color(0xFF0D1527),
              ],
            ),
            borderRadius: BorderRadius.circular(16),
            border: Border.all(color: accentColor.withValues(alpha: 0.6)),
          ),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Text(
                isHealthy ? 'ESTADO PULMONAR: NORMAL' : 'HALLAZGO PATOLÓGICO DETECTADO',
                style: TextStyle(
                  fontSize: 11,
                  fontWeight: FontWeight.bold,
                  color: accentColor,
                ),
              ),
              const SizedBox(height: 6),
              Text(
                predictedClass,
                style: const TextStyle(
                  fontSize: 22,
                  fontWeight: FontWeight.bold,
                  color: Colors.white,
                ),
              ),
              const SizedBox(height: 4),
              Text(
                'Confianza diagnóstica: ${confidence.toStringAsFixed(2)}%',
                style: const TextStyle(fontSize: 12, color: Color(0xFFCBD5E1)),
              ),
            ],
          ),
        ),
        const SizedBox(height: 18),
        const Text(
          'Distribución de probabilidad por patología:',
          style: TextStyle(fontSize: 13, fontWeight: FontWeight.bold, color: Color(0xFF94A3B8)),
        ),
        const SizedBox(height: 10),
        ...probs.entries.map((entry) {
          final pct = entry.value;
          return Padding(
            padding: const EdgeInsets.only(bottom: 8),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.stretch,
              children: [
                Row(
                  mainAxisAlignment: MainAxisAlignment.spaceBetween,
                  children: [
                    Text(entry.key, style: const TextStyle(fontSize: 12, color: Colors.white)),
                    Text('${(pct * 100).toStringAsFixed(2)}%', style: const TextStyle(fontSize: 12, color: Color(0xFF00F0FF))),
                  ],
                ),
                const SizedBox(height: 4),
                ClipRRect(
                  borderRadius: BorderRadius.circular(4),
                  child: LinearProgressIndicator(
                    value: pct.clamp(0.0, 1.0),
                    backgroundColor: const Color(0xFF1E293B),
                    valueColor: const AlwaysStoppedAnimation<Color>(Color(0xFF00F0FF)),
                    minHeight: 6,
                  ),
                ),
              ],
            ),
          );
        }),
      ],
    );
  }
}

class _TagBadge extends StatelessWidget {
  final String label;
  const _TagBadge({required this.label});

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 7, vertical: 2),
      decoration: BoxDecoration(
        color: const Color(0x26A855F7),
        borderRadius: BorderRadius.circular(8),
        border: Border.all(color: const Color(0x4DA855F7)),
      ),
      child: Text(
        label,
        style: const TextStyle(fontSize: 10, color: Color(0xFFD8B4FE)),
      ),
    );
  }
}
