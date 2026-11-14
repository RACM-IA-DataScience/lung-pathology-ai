# AI in HealthCare - Chest Disease Detection

![Python](https://img.shields.io/badge/Python-3.9+-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.21.0-orange)
![Keras](https://img.shields.io/badge/Keras-3.14.1-red)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.8.0-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-1.57.0-red)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange)

Este proyecto implementa un modelo de inteligencia artificial basado en Deep Learning para la clasificación y detección de enfermedades torácicas (Neumonía Bacteriana, Neumonía Viral y COVID-19) o pulmones sanos a partir de radiografías de tórax (X-Rays). 

El modelo soporta arquitecturas modernas (**DenseNet-121** por defecto, **ConvNeXt-Tiny** y **MobileNetV2**) mediante Transfer Learning con `tf.data`, junto con una aplicación web interactiva en **Streamlit** con explicabilidad visual **Grad-CAM / Grad-CAM++** en tiempo real.

## 🚀 Características Principales
- **Clasificación Multiclase Dinámica:** `Bacterial Pneumonia`, `Healthy`, `Viral Pneumonia` y `covid-19` (adaptable dinámicamente al agregar nuevas carpetas de patologías).
- **Backbones Especializados:** Selección entre **DenseNet-121** (estándar clínico en CXR), **ConvNeXt-Tiny** y **MobileNetV2** desde [`src/model_utils.py`](src/model_utils.py).
- **Fine-Tuning de 2 Fases + Label Smoothing:** Descongelamiento del bloque superior (manteniendo `BatchNormalization` congelada) y suavizado de etiquetas (`0.05`) para mejor calibración clínica.
- **Saneamiento y Split Sin Fugas (`GroupShuffleSplit`):** Deduplicación por linaje, hash exacto SHA-256 y similitud perceptual (`dHash-256` + MAE) + partición agrupada por paciente (`70% train / 15% val / 15% test`).
- **Ingesta de Datasets Externos:** Soporte en [`scripts/ingestar_datasets.py`](scripts/ingestar_datasets.py) para **COVID-19 Radiography Database** (`COVID`, `Normal` adultos RSNA y `Lung_Opacity` RSNA), **COVID-QU-Ex** y **RSNA Pneumonia** (DICOM).
- **Data Augmentation con `tf.data`:** Pipeline optimizado con transformaciones en grafo de TensorFlow preservando la lateralidad anatómica.
- **Interfaz Web (Streamlit) + Grad-CAM++:** Visualización lado a lado de placa original y mapa de calor de atención pulmonar, estimación de zonas anatómicas activadas y alerta ante baja confianza (`< 60%`).

## 📋 Estructura del Proyecto

El repositorio tiene la siguiente estructura de archivos principal:

```
├── src/
│   ├── app.py                    # Interfaz Streamlit + Grad-CAM / Grad-CAM++ interactivo
│   ├── model_utils.py            # Backbones (DenseNet121, ConvNeXtTiny, MobileNetV2) y Grad-CAM
│   ├── train_model.py            # Entrenamiento en 2 fases con tf.data y label smoothing
│   └── evaluate_model.py         # Evaluación en test, métricas JSON, matriz y mosaico Grad-CAM++
├── scripts/
│   ├── ingestar_datasets.py      # Descarga/ingesta de COVID-19 Radiography, COVID-QU-Ex y RSNA
│   ├── procesar_dataset.py       # Deduplicación (linaje + SHA-256) y estandarización 224x224 RGB
│   └── reorganizar_dataset.py    # Partición 70/15/15 agrupada por paciente + auditoría sin fugas
├── config/
│   ├── mapeo_clases.json         # Diccionario dinámico de clases e índices
│   └── model_config.json         # Metadatos del backbone activo y capa objetivo Grad-CAM
├── models/
│   └── modelo_chest_disease.keras # Modelo entrenado
├── results/
│   ├── matriz_confusion.png      # Matriz de confusión (conteos y recall %)
│   ├── gradcam_auditoria.png     # Mosaico de explicabilidad Grad-CAM++ en test
│   └── metricas_evaluacion.json  # Reporte estructurado de métricas
├── data/                         # Generado localmente (ignorado en git)
│   ├── downloads/                # ZIPs/carpetas externas descargadas
│   ├── raw/                      # Datasets crudos de entrada (solo lectura)
│   ├── processed/{train,val,test}/ # Imágenes limpias 224x224 sin fugas
│   └── dataset_manifest.csv      # Manifiesto auditable por paciente y hash
├── documentacion_proyecto.md     # Documentación extensa del proceso y optimización del modelo
├── DOCUMENTACION_TECNICA.md      # Resumen técnico y diagrama de flujo
└── requirements.txt              # Dependencias de Python necesarias para ejecutar el proyecto
```

*(Nota: Las carpetas de datos `data/`, `Training Data/` y `Testing Data/` están excluidas en `.gitignore` para no sobrecargar el repositorio).*

## ⚙️ Instalación y Configuración

Sigue estos pasos para ejecutar el proyecto localmente.

### 1. Clonar el repositorio
```bash
git clone https://github.com/RobbeAlex/AI-in-HealthCare-Chest-Disease-Detection
cd AI-in-HealthCare-Chest-Disease-Detection
```

### 2. Crear un entorno virtual (Recomendado)
Es altamente recomendado crear un entorno virtual para aislar las dependencias.
```bash
python -m venv .venv
# En Windows:
.venv\Scripts\activate
# En Linux/Mac:
source .venv/bin/activate
```

### 3. Instalar dependencias
```bash
pip install -r requirements.txt
```

## 💻 Uso

Para arrancar la interfaz web y empezar a probar el modelo con tus propias radiografías, ejecuta:

```bash
streamlit run src/app.py
```
Esto abrirá automáticamente una pestaña en tu navegador web donde podrás interactuar con el sistema de predicción.

## 🧠 Entrenamiento y Evaluación (Opcional)

Si deseas ampliar las 4 clases actuales (recomendado para corregir la baja muestra de `covid-19` y el sesgo pediátrico), reentrenar o evaluar:

1. **Ingestar datasets externos (opcional, recomendado):**
   - Con Kaggle CLI configurado: `python scripts/ingestar_datasets.py --dataset covid19-radiography --download`
   - O colocando los ZIPs en `data/downloads/<dataset>/` y ejecutando sin `--download`.
   - También soporta `--dataset covid-qu-ex` y `--dataset rsna-pneumonia`.
2. **Deduplicar y estandarizar imágenes:** `python scripts/procesar_dataset.py`
3. **Particionar por paciente (`70/15/15`) y verificar cero fugas:** `python scripts/reorganizar_dataset.py`
4. **Entrenar (`DenseNet121` por defecto o `ConvNeXtTiny`):**
   - `python src/train_model.py --backbone densenet121`
   - `python src/train_model.py --backbone convnext_tiny`
5. **Evaluar + auditoría Grad-CAM++:** `python src/evaluate_model.py` (actualiza `results/matriz_confusion.png`, `results/metricas_evaluacion.json` y `results/gradcam_auditoria.png`).

## 📈 Resultados (DenseNet-121 sobre 2,078 placas de test sin fugas)

Tras eliminar duplicados recompresos (`73` exactos + `1,096` perceptuales), ampliar a **13,906 radiografías limpias** (mezcla pediátrica y de adultos RSNA/BIMCV/SIRM) y migrar a **DenseNet-121 + `tf.data` + Label Smoothing (`0.05`)**:

- **Accuracy global en Test (`n = 2,078`):** **84.41%** (`ROC-AUC Macro OvR: 0.9663`)
- **`covid-19` (`n = 507`):** Precisión **96.24%**, Recall **80.87%**, F1 **87.89%**
- **`Healthy` (`n = 752`):** Precisión **87.61%**, Recall **94.95%**, F1 **91.13%**
- **`Bacterial Pneumonia` (`n = 593`):** Precisión **78.69%**, Recall **79.09%**, F1 **78.89%**
- **`Viral Pneumonia` (`n = 226`):** Precisión **66.80%**, Recall **71.24%**, F1 **68.95%**
- **Generalización en adultos inéditos RSNA (`n = 400` fuera del dataset):** `71.0%` de detección en neumonía de adultos (frente a `5.5%` del modelo previo) y `98.0%` en sanos de adultos.

Consulta el desglose completo y auditoría visual en [documentacion_proyecto.md](documentacion_proyecto.md), [results/metricas_evaluacion.json](results/metricas_evaluacion.json) y [results/gradcam_auditoria.png](results/gradcam_auditoria.png).

## 📝 Licencia / Autoría
Desarrollado por Roberto Alejandro Chagra Martínez
