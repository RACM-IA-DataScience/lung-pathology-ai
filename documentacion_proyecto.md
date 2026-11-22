# Documentación del Proyecto de Detección de Enfermedades Torácicas

## 1. Dataset Elegido

**Fuentes de Datos Curadas (13,906 radiografías limpias tras deduplicación):** Se fusionaron cohortes pediátricas y de adultos garantizando 0% de fugas y sin clases redundantes:
1. *Chest X-Ray Images - Pneumonia* (Kermany / Paul Mooney, cohorte pediátrica con identificación por paciente `personID` / `IM-ID`)
2. *COVID-19 Radiography Database* (Tawsifur Rahman — `COVID-19` multi-cohorte BIMCV/SIRM/Eurorad/Cohen, `Normal` pediátrico+RSNA y `Lung_Opacity` de adultos RSNA; se excluyó su subconjunto `Viral Pneumonia` por ser 100% redundante con Kermany)

**Nota clínica sobre mapeo de adultos:** Para corregir el sesgo pediátrico del baseline (que fallaba en el 94.5% de las neumonías de adultos), se incorporaron 1,400 placas de adultos `RSNA Lung_Opacity` (opacidades pulmonares por neumonía adquirida en comunidad, predominantemente bacteriana/consolidadora en adultos) dentro de `Bacterial Pneumonia`, manteniendo a Kermany como fuente de diferenciación microbiológica pediátrica bacteriana vs. viral.

**Distribución final particionada por paciente (`data/processed/`, 0 colisiones inter-clase):**
- `Bacterial Pneumonia`: **4,045** (`train=2,840`, `val=612`, `test=593` | 2,886 pacientes únicos)
- `Healthy`: **5,013** (`train=3,509`, `val=752`, `test=752` | 5,009 pacientes únicos)
- `Viral Pneumonia`: **1,470** (`train=1,025`, `val=219`, `test=226` | 1,222 pacientes únicos)
- `covid-19`: **3,378** (`train=2,364`, `val=507`, `test=507` | 3,378 pacientes únicos)

**Preprocesamiento, Deduplicación y Auditoría de Imágenes (`scripts/procesar_dataset.py`, `scripts/reorganizar_dataset.py`):**
- Se separaron las carpetas de entrada de solo lectura (`data/raw`, `Training Data`, `Testing Data`) de las salidas limpias (`data/clean` → `data/processed/{train,val,test}`) para impedir auto-duplicación (`kaggle_kaggle_...`) y sobreescritura accidental.
- Se deduplicaron archivos por linaje de nombre, por hash criptográfico **SHA-256** de los píxeles normalizados (`224x224 RGB`) y por firma perceptual **`dHash-256` + MAE `32x32`** (`73` duplicados exactos y `1,096` clones recompresos PNG↔JPEG eliminados).
- Se forzó conversión al espacio `RGB`, redimensionamiento a `224x224` píxeles con interpolación `LANCZOS` y exportación en `JPEG` (calidad 95, evitando recomprimir archivos que ya cumplen ese estándar).
- El particionamiento en `train` (70%), `val` (15%) y `test` (15%) se realiza con `GroupShuffleSplit` agrupado por identificador de paciente (`personID` / `IM-ID` / `rahman-ID`) y se valida mediante `data/dataset_manifest.csv` para garantizar **0% de *data leakage*** entre particiones.

---

## 2. Documentación del Proceso de Optimización del Modelo

### 2.1 Ajustes de Arquitectura (`src/model_utils.py` + Transfer Learning)
- **Backbones Soportados:** Se centralizó la creación de arquitecturas en [`src/model_utils.py`](src/model_utils.py) mediante Functional API para exponer la capa espacial `gradcam_target`:
  - **DenseNet-121** (`densenet121`, predeterminado): estándar clínico en radiografía torácica (CheXNet) con denso cabezal de `256` unidades.
  - **ConvNeXt-Tiny** (`convnext_tiny`): arquitectura convolucional moderna de alto desempeño.
  - **MobileNetV2** (`mobilenet_v2`): alternativa ligera compatible con versiones anteriores.
- **Arquitectura de Salida Personalizada:**
  - Capa `Activation('linear', name='gradcam_target')` para asegurar compatibilidad universal con **Grad-CAM / Grad-CAM++**.
  - Capa `GlobalAveragePooling2D`.
  - Capa densa oculta (`256` neuronas en DenseNet/ConvNeXt; `128` en MobileNetV2) con `ReLU` y regularización `L2 (1e-4)`.
  - Capa `Dropout (0.4)` (40%) para prevenir sobreajuste (*overfitting*).
  - Capa de salida dinámica `Dense(num_classes, activation='softmax')` sincronizada con `config/mapeo_clases.json` y `config/model_config.json`.

### 2.2 Balanceo y Aumento de Datos (`tf.data` + Data Augmentation)
- **Corrección de Desbalanceo de Clases:** Se calcula `compute_class_weight='balanced'` sobre la distribución real y deduplicada del conjunto `data/processed/train`.
- **Data Augmentation:** Implementado directamente en `tf.data` + capas nativas de Keras sobre `[0, 255]` (preservando lateralidad anatómica derecha/izquierda) antes del `preprocess_input` específico de cada backbone:
  - Rotaciones aleatorias leves (`±7°`, `RandomRotation`).
  - Zoom aleatorio (`±8%`, `RandomZoom`).
  - Desplazamientos horizontales y verticales (`±5%`, `RandomTranslation`).
  - Variación de contraste (`±10%`, `RandomContrast`).

### 2.3 Estrategia de Entrenamiento (Two-Phase Fine-Tuning + Label Smoothing)
1. **Fase 1: Entrenamiento del Clasificador Superior (Feature Extraction)**
   - Backbone congelado (`trainable = False`).
   - **Pérdida y Métricas:** `CategoricalCrossentropy(label_smoothing=0.05)`, `accuracy`, `Precision`, `Recall` y `AUC`.
   - **Optimizador:** Adam (`learning_rate = 1e-3`, `batch_size = 64`).
   - **Épocas:** 3 (con `EarlyStopping`, paciencia 3 sobre `val_loss`).

2. **Fase 2: Ajuste Fino Avanzado (Fine-Tuning)**
   - Se descongelan las últimas capas del bloque superior (`60` en DenseNet-121 — `43` capas activas con `17 Conv2D`, `40` en ConvNeXt-Tiny, `20` en MobileNetV2) manteniendo todas las `BatchNormalization` congeladas.
   - **Optimizador:** Adam (`learning_rate = 4e-5`).
   - **Épocas:** 8, con `EarlyStopping` (paciencia 5), `ReduceLROnPlateau` (factor `0.5`, paciencia 2) y `ModelCheckpoint` (`save_best_only=True` sobre `val_loss`) guardando en `models/modelo_chest_disease.keras`.

### 2.4 Evaluación y Explicabilidad Visual (Grad-CAM++)
- **Evaluación Estrita Sin Barajado:** [`src/evaluate_model.py`](src/evaluate_model.py) evalúa sobre `data/processed/test` con `shuffle=False`, detecta automáticamente el backbone en `config/model_config.json`, calcula `ROC-AUC Macro` y exporta `results/metricas_evaluacion.json`, `results/matriz_confusion.png` y el mosaico `results/gradcam_auditoria.png`.
- **Integración Web con Grad-CAM / Grad-CAM++:** [`src/app.py`](src/app.py) muestra lado a lado la radiografía original y el mapa de calor superpuesto (con control de opacidad, selector de paleta y selector de clase a explicar), además de estimar la activación porcentual en 6 zonas radiológicas y alertar cuando la confianza es menor al 60%.

---

## 3. Resultados Verificados en Test Sin Fugas y Generalización en Adultos

Sobre el conjunto de prueba de **2,078 radiografías** (agrupadas por paciente y auditadas sin duplicados ni fugas):

| Clase | Precisión | Recall | F1-Score | Soporte |
|---|---:|---:|---:|---:|
| `Bacterial Pneumonia` | 0.7869 | 0.7909 | 0.7889 | 593 |
| `Healthy` | 0.8761 | 0.9495 | 0.9113 | 752 |
| `Viral Pneumonia` | 0.6680 | 0.7124 | 0.6895 | 226 |
| `covid-19` | 0.9624 | 0.8087 | 0.8789 | 507 |
| **Global / Macro** | **0.8441 (Acc)** | **0.8154 (Macro)** | **0.8171 (Macro)** | **2,078** |

- **ROC-AUC Macro (OvR):** **0.9663** (supera al baseline `MobileNetV2` de `0.9642` sobre un test 2× más grande y multi-cohorte).
- **Prueba en adultos inéditos RSNA (`n = 400` imágenes fuera de train/val/test):** la detección de neumonía en adultos (`Lung_Opacity`) subió de **5.5% (`11/200`)** a **71.0% (`142/200`)**, reduciendo los falsos positivos hacia `covid-19` de **18.5% (`37/200`)** a **0.5% (`1/200`)**, y los sanos adultos alcanzaron **98.0% (`196/200`)**.
- **Limitación conocida:** `Viral Pneumonia` continúa proviniendo de cohorte pediátrica (Kermany), por lo que la diferenciación etiológica *bacteriana vs. viral* en adultos sigue siendo la principal línea de mejora para pasos futuros.

---

## 4. Autoría y Créditos
- **Creador del Modelo e Investigador:** Roberto Alejandro Chagra Martínez
- **Firma:** RACM (Artificial Intelligence & Data Science)
- **Interfaz Interactiva:** Streamlit + Grad-CAM++ con identidad visual personalizada y paleta ciberpunk médica.
