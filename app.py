"""AI Chest Disease Detection - Gradio Interface para Hugging Face Spaces.

Creador del Modelo: Roberto Alejandro Chagra Martínez
Disciplina: Artificial Intelligence & Data Science (RACM AI)
Arquitectura: DenseNet-121 con Interpretabilidad Grad-CAM / Grad-CAM++
"""

import base64
import json
from pathlib import Path

import gradio as gr
import numpy as np
from PIL import Image
import tensorflow as tf

# En Hugging Face ZeroGPU / Cloud, ejecutar en CPU para evitar bloqueo de VRAM
try:
    tf.config.set_visible_devices([], "GPU")
except Exception:
    pass

from src.model_utils import (
    IMG_SIZE,
    compute_gradcam_heatmap,
    get_backbone_specs,
    load_model_metadata,
    overlay_gradcam,
    preprocess_pil_image,
)

BASE_DIR = Path(__file__).resolve().parent

# Cargar mapeo de clases
CONFIG_PATH = BASE_DIR / "config" / "mapeo_clases.json"
if CONFIG_PATH.exists():
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        CLASES_DICT = json.load(f)
else:
    CLASES_DICT = {"0": "Bacterial Pneumonia", "1": "Healthy", "2": "Viral Pneumonia", "3": "covid-19"}

# Cargar modelo entrenado
MODEL_PATH = BASE_DIR / "models" / "modelo_chest_disease.keras"
model = None
MODEL_META = None

def load_keras_model_safe(model_path):
    try:
        return tf.keras.models.load_model(model_path)
    except Exception as exc:
        import zipfile, tempfile, shutil
        temp_dir = Path(tempfile.mkdtemp())
        clean_model_file = temp_dir / "clean.keras"
        def _clean(obj):
            if isinstance(obj, dict):
                obj.pop("quantization_config", None)
                for v in obj.values():
                    _clean(v)
            elif isinstance(obj, list):
                for v in obj:
                    _clean(v)
        with zipfile.ZipFile(model_path, 'r') as zin, zipfile.ZipFile(clean_model_file, 'w') as zout:
            for item in zin.infolist():
                d = zin.read(item.filename)
                if item.filename == 'config.json':
                    c = json.loads(d.decode('utf-8'))
                    _clean(c)
                    d = json.dumps(c).encode('utf-8')
                zout.writestr(item, d)
        m = tf.keras.models.load_model(clean_model_file)
        shutil.rmtree(temp_dir, ignore_errors=True)
        return m

if MODEL_PATH.exists():
    model = load_keras_model_safe(MODEL_PATH)
    MODEL_META = load_model_metadata(BASE_DIR / "config", model=model)

backbone = MODEL_META.get("backbone", "densenet121") if MODEL_META else "densenet121"
img_size = tuple(MODEL_META.get("img_size", list(IMG_SIZE))) if MODEL_META else IMG_SIZE

# Logotipo oficial RACM
LOGO_PATH = BASE_DIR / "assets" / "racm_logo.jpg"
logo_b64 = ""
if LOGO_PATH.exists():
    with open(LOGO_PATH, "rb") as f:
        logo_b64 = f"data:image/jpeg;base64,{base64.b64encode(f.read()).decode('utf-8')}"

# Soporte nativo para Hugging Face ZeroGPU
try:
    import spaces
    has_spaces = True
except ImportError:
    has_spaces = False

def _run_prediction(image, method, alpha, colormap):
    if image is None:
        return None, {}, "⚠️ Por favor, sube una radiografía de tórax (CXR)."

    if model is None:
        return None, {}, "❌ Error: El archivo del modelo neuronal no se encuentra cargado."

    if not isinstance(image, Image.Image):
        image = Image.fromarray(image)

    # Inferencia y cálculo de tensores Grad-CAM
    tensor_imagen = preprocess_pil_image(image, backbone=backbone, img_size=img_size)
    heatmap, explained_idx, predicciones = compute_gradcam_heatmap(
        model=model,
        preprocessed_tensor=tensor_imagen,
        class_index=None,
        method=method,
    )

    overlay_img = overlay_gradcam(image, heatmap, alpha=alpha, cmap_name=colormap)

    indice_pred = int(np.argmax(predicciones))
    clase_detectada = CLASES_DICT[str(indice_pred)]
    confianza = float(predicciones[indice_pred] * 100.0)
    es_sano = clase_detectada.lower() in {"healthy", "normal"}

    # Formatear diccionario de probabilidades para componente Label
    probs_dict = {
        CLASES_DICT[str(i)]: float(predicciones[i])
        for i in range(len(CLASES_DICT))
    }

    badge_texto = "ESTADO PULMONAR: NORMAL" if es_sano else "HALLAZGO PATOLÓGICO DETECTADO"
    badge_color = "#10B981" if es_sano else "#F43F5E"

    summary_html = f"""
    <div style="background: linear-gradient(135deg, rgba(16,24,44,0.9) 0%, rgba(13,21,39,0.95) 100%);
                padding: 18px 22px; border-radius: 14px; border: 1px solid {badge_color};">
        <div style="font-size: 0.85rem; font-weight: 700; color: {badge_color}; letter-spacing: 0.5px;">
            {badge_texto}
        </div>
        <div style="font-size: 1.8rem; font-weight: 800; color: #ffffff; margin: 4px 0;">
            {clase_detectada}
        </div>
        <div style="font-size: 0.95rem; color: #cbd5e1;">
            Confianza diagnóstica: <strong style="color: #00f0ff;">{confianza:.2f}%</strong>
        </div>
        <div style="font-size: 0.8rem; color: #94a3b8; margin-top: 6px;">
            Interpretabilidad: <strong>{method.upper()}</strong> | Paleta térmica: <strong>{colormap}</strong> | Opacidad: <strong>{alpha:.2f}</strong>
        </div>
    </div>
    """
    return overlay_img, probs_dict, summary_html

if has_spaces:
    @spaces.GPU(duration=45)
    def predict_and_explain(image, method, alpha, colormap):
        return _run_prediction(image, method, alpha, colormap)
else:
    def predict_and_explain(image, method, alpha, colormap):
        return _run_prediction(image, method, alpha, colormap)

header_html = f"""
<div style="background: radial-gradient(circle at 20% 20%, #0d1527 0%, #070b14 70%);
            border: 1px solid rgba(0,240,255,0.25); border-radius: 18px; padding: 22px; margin-bottom: 20px;
            box-shadow: 0 8px 32px rgba(0,0,0,0.4);">
    <div style="display: flex; align-items: center; gap: 18px; flex-wrap: wrap;">
        <div style="width: 76px; height: 76px; border-radius: 50%; padding: 3px;
                    background: linear-gradient(135deg, #00f0ff, #a855f7); flex-shrink: 0;">
            <img src="{logo_b64}" style="width: 100%; height: 100%; border-radius: 50%; object-fit: cover; display: block;" />
        </div>
        <div style="flex: 1; min-width: 250px;">
            <div style="font-size: 0.75rem; color: #00f0ff; font-weight: 700; text-transform: uppercase; letter-spacing: 1px;">
                ● Sistema Inteligente de Diagnóstico Torácico
            </div>
            <div style="font-size: 1.6rem; font-weight: 800; color: #ffffff; line-height: 1.2;">
                Detección de Enfermedades Pulmonares con IA
            </div>
            <div style="font-size: 0.85rem; color: #94a3b8; margin-top: 4px;">
                Creador del Modelo: <strong style="color: #ffffff;">Roberto Alejandro Chagra Martínez</strong> • Disciplina: <strong style="color: #a855f7;">Artificial Intelligence & Data Science</strong> • <span style="background: rgba(0,240,255,0.15); color: #00f0ff; padding: 2px 8px; border-radius: 6px; font-weight: bold;">RACM</span>
            </div>
        </div>
    </div>
</div>
"""

custom_theme = gr.themes.Default(
    primary_hue="cyan",
    secondary_hue="purple",
    neutral_hue="slate",
).set(
    body_background_fill="#070B14",
    body_background_fill_dark="#070B14",
    block_background_fill="#0D1527",
    block_background_fill_dark="#0D1527",
    block_border_color="rgba(0, 240, 255, 0.2)",
    block_border_color_dark="rgba(0, 240, 255, 0.2)",
)

with gr.Blocks(theme=custom_theme, title="AI Chest Disease Detection - RACM") as demo:
    gr.HTML(header_html)

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 📂 Entrada Radiológica")
            input_image = gr.Image(
                type="pil",
                label="Arrastra o selecciona una radiografía de tórax (CXR)",
                sources=["upload", "clipboard"],
            )

            with gr.Accordion("⚙️ Configuración Visual de Grad-CAM", open=True):
                method_radio = gr.Radio(
                    choices=["gradcam++", "gradcam"],
                    value="gradcam++",
                    label="Método de Explicabilidad",
                    info="Grad-CAM++ ofrece mayor precisión focal multiobjeto.",
                )
                alpha_slider = gr.Slider(
                    minimum=0.15,
                    maximum=0.75,
                    value=0.42,
                    step=0.05,
                    label="Opacidad del mapa térmico",
                )
                cmap_dropdown = gr.Dropdown(
                    choices=["jet", "inferno", "turbo"],
                    value="jet",
                    label="Paleta térmica (Colormap)",
                )

            analyze_btn = gr.Button("⚡ Analizar Radiografía y Generar Grad-CAM", variant="primary")

        with gr.Column(scale=1):
            gr.Markdown("### 🎯 Mapa de Atención Grad-CAM")
            output_overlay = gr.Image(type="pil", label="Foco de Atención Neuronal Superpuesto")

        with gr.Column(scale=1):
            gr.Markdown("### 📊 Diagnóstico y Probabilidades")
            output_summary = gr.HTML()
            output_label = gr.Label(num_top_classes=4, label="Distribución de Probabilidad")

    analyze_btn.click(
        fn=predict_and_explain,
        inputs=[input_image, method_radio, alpha_slider, cmap_dropdown],
        outputs=[output_overlay, output_label, output_summary],
    )

    gr.Markdown(
        """
        ---
        <div style="text-align: center; font-size: 0.8rem; color: #64748b;">
            Desarrollado y calibrado por <strong>Roberto Alejandro Chagra Martínez</strong> | Artificial Intelligence & Data Science | RACM AI
        </div>
        """
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
