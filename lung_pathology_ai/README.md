# 🫁 Lung Pathology AI - Cliente Multiplataforma en Flutter

Aplicación de diagnóstico radiológico con Inteligencia Artificial e interpretabilidad biológica en tiempo real mediante mapas de atención **Grad-CAM / Grad-CAM++**.

**Creador del Modelo:** Roberto Alejandro Chagra Martínez  
**Disciplina:** Artificial Intelligence & Data Science  
**Marca / Identidad:** RACM AI  

---

## 📋 Requisitos del Sistema

### 1. Requisitos de Flutter
- **Flutter SDK:** 3.24+ (o 3.50+)
- **Dart SDK:** 3.5+
- **Plataformas soportadas:** Windows Desktop, Web (Chrome / Servidor Web), Android, iOS.

### 2. Dependencias de Flutter (`pubspec.yaml`)
- `file_picker`: Selector de archivos radiológicos locales (JPG, PNG).
- `http`: Comunicación con el motor de inferencia neuronal.
- `cupertino_icons`: Iconografía clínica y de navegación.

### 3. Requisitos de Inferencia Python (`requirements-flutter.txt`)
Para calcular los gradientes Grad-CAM++ y ejecutar el modelo de TensorFlow (`models/modelo_chest_disease.keras`):
```text
tensorflow>=2.16.0
keras>=3.0.0
numpy>=1.24.0,<3.0.0
Pillow>=9.5.0
matplotlib>=3.7.0
absl-py>=2.0.0
```

---

## 🚀 Guía de Ejecución

### Paso 1: Iniciar el Motor de Inferencia Neuronal (Backend)
Desde la raíz del repositorio principal:
```powershell
.\.venv\Scripts\python.exe src/api.py
```
*(El servidor quedará a la escucha en `http://127.0.0.1:8000` y `http://0.0.0.0:8000`)*.

### Paso 2: Ejecutar la Aplicación Flutter

Navega al directorio de Flutter:
```powershell
cd lung_pathology_ai
```

#### A. Como Aplicación de Escritorio Nativa (Windows Desktop)
```powershell
flutter run -d windows
```

#### B. Como Aplicación Web (Navegador)
```powershell
flutter run -d web-server --web-port=8080
```
Y abre en tu navegador: **http://localhost:8080**

#### C. Compilar APK para Teléfonos Android
```powershell
flutter build apk --release
```
El archivo instalador se generará en:
`build/app/outputs/flutter-apk/app-release.apk`
