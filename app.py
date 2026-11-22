# IMPORTANTE PARA HUGGING FACE ZEROGPU:
# "import spaces" debe ser la primera línea absoluta antes de cualquier librería con bindings de CUDA o TensorFlow.
try:
    import spaces
    has_spaces = True
except ImportError:
    has_spaces = False

import base64
import json
from pathlib import Path

import gradio as gr
import numpy as np
from PIL import Image
import tensorflow as tf

# Ejecutar inferencia en CPU para máxima estabilidad y evitar bloqueos de CUDA en ZeroGPU
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
    summarize_activation_zones,
)

BASE_DIR = Path(__file__).resolve().parent

# Cargar mapeo de clases
CONFIG_PATH = BASE_DIR / "config" / "mapeo_clases.json"
if CONFIG_PATH.exists():
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        CLASES_DICT = json.load(f)
else:
    CLASES_DICT = {"0": "Bacterial Pneumonia", "1": "Healthy", "2": "Viral Pneumonia", "3": "covid-19"}

# Mapeo inverso de nombre de clase a índice
NAME_TO_IDX = {name: int(k) for k, name in CLASES_DICT.items()}

# Ruta del modelo entrenado
MODEL_PATH = BASE_DIR / "models" / "modelo_chest_disease.keras"

def load_keras_model_safe(model_path):
    try:
        return tf.keras.models.load_model(model_path)
    except Exception:
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

_cached_model = None
_cached_meta = None

def get_model():
    global _cached_model, _cached_meta
    if _cached_model is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"No se encontró el modelo en {MODEL_PATH}")
        try:
            gpus = tf.config.list_physical_devices("GPU")
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except Exception:
            pass
        _cached_model = load_keras_model_safe(MODEL_PATH)
        _cached_meta = load_model_metadata(BASE_DIR / "config", model=_cached_model)
    return _cached_model, _cached_meta

# Logotipo oficial RACM
LOGO_PATH = BASE_DIR / "assets" / "racm_logo.jpg"
logo_b64 = ""
if LOGO_PATH.exists():
    with open(LOGO_PATH, "rb") as f:
        logo_b64 = f"data:image/jpeg;base64,{base64.b64encode(f.read()).decode('utf-8')}"

def _run_prediction(image, method, alpha, colormap, target_class_choice):
    if image is None:
        return None, "", {}, "⚠️ Por favor, sube una radiografía de tórax (CXR)."

    try:
        model, meta = get_model()
        backbone = meta.get("backbone", "densenet121") if meta else "densenet121"
        img_size = tuple(meta.get("img_size", list(IMG_SIZE))) if meta else IMG_SIZE

        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)

        # Resolver clase a explicar (automática vs selección manual)
        if target_class_choice == "Predicción principal (automática)" or not target_class_choice:
            target_class_idx = None
        else:
            target_class_idx = NAME_TO_IDX.get(target_class_choice, None)

        # Inferencia y cálculo de tensores Grad-CAM
        tensor_imagen = preprocess_pil_image(image, backbone=backbone, img_size=img_size)
        heatmap, explained_idx, predicciones = compute_gradcam_heatmap(
            model=model,
            preprocessed_tensor=tensor_imagen,
            class_index=target_class_idx,
            method=method,
        )

        overlay_img = overlay_gradcam(image, heatmap, alpha=alpha, cmap_name=colormap)

        # Segmentación anatómica de la atención por zonas pulmonares
        zone_summary = summarize_activation_zones(heatmap)
        top_zone = zone_summary.get("top_zone", "Sin focalización clara")
        top_share = zone_summary.get("top_share", 0.0)
        zones_data = zone_summary.get("zones", {})
        sorted_zones = sorted(zones_data.items(), key=lambda kv: kv[1], reverse=True)

        indice_pred = int(np.argmax(predicciones))
        clase_detectada = CLASES_DICT[str(indice_pred)]
        clase_explicada = CLASES_DICT[str(explained_idx)]
        confianza = float(predicciones[indice_pred] * 100.0)
        es_sano = clase_detectada.lower() in {"healthy", "normal"}

        # Formatear diccionario de probabilidades para componente Label
        probs_dict = {
            CLASES_DICT[str(i)]: float(predicciones[i])
            for i in range(len(CLASES_DICT))
        }

        badge_texto = "ESTADO PULMONAR: NORMAL" if es_sano else "HALLAZGO PATOLÓGICO DETECTADO"
        badge_color = "#10B981" if es_sano else "#F43F5E"

        # Tarjeta HTML de Diagnóstico Principal
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
                Explicando clase: <strong style="color: #a855f7;">{clase_explicada}</strong> ({method.upper()}) | Paleta: <strong>{colormap}</strong> | Opacidad: <strong>{alpha:.2f}</strong>
            </div>
        </div>
        """

        # Tarjeta HTML de Segmentación por Zonas Anatómicas con barras de masa térmica
        zones_html = f"""
        <div style="background: rgba(13,21,39,0.9); border: 1px solid rgba(0,240,255,0.3); border-radius: 14px; padding: 16px 18px; margin-top: 14px;">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                <span style="font-size: 0.9rem; font-weight: bold; color: #00f0ff;">📍 Segmentación de Atención Pulmonar:</span>
                <span style="font-size: 0.72rem; background: rgba(0,240,255,0.15); color: #00f0ff; padding: 2px 8px; border-radius: 6px; font-weight: bold;">{method.upper()}</span>
            </div>
            <div style="font-size: 0.82rem; color: #cbd5e1; margin-bottom: 12px;">
                Región dominante: <strong style="color: #ffffff;">{top_zone}</strong> (<span style="color: #00f0ff; font-weight: bold;">{top_share:.1f}%</span> de masa térmica)
            </div>
        """
        for z_name, z_pct in sorted_zones:
            zones_html += f"""
            <div style="margin-bottom: 8px;">
                <div style="display: flex; justify-content: space-between; font-size: 0.78rem; color: #e2e8f0; margin-bottom: 3px;">
                    <span>{z_name}</span>
                    <span style="color: #00f0ff; font-weight: bold;">{z_pct:.1f}%</span>
                </div>
                <div style="width: 100%; height: 6px; background: rgba(255,255,255,0.08); border-radius: 3px; overflow: hidden;">
                    <div style="width: {min(100.0, z_pct)}%; height: 100%; background: linear-gradient(90deg, #00f0ff, #a855f7); border-radius: 3px;"></div>
                </div>
            </div>
            """
        zones_html += "</div>"

        return overlay_img, zones_html, probs_dict, summary_html

    except Exception as exc:
        import traceback
        err_detail = traceback.format_exc()
        err_html = f"""
        <div style="background: rgba(244,63,94,0.15); border: 1px solid #f43f5e; border-radius: 12px; padding: 16px;">
            <div style="color: #f43f5e; font-weight: bold; font-size: 1rem;">⚠️ Error durante la inferencia: {exc}</div>
            <pre style="color: #ffb4c2; font-size: 11px; margin-top: 8px; overflow-x: auto; white-space: pre-wrap;">{err_detail}</pre>
        </div>
        """
        return None, "", {}, err_html

if has_spaces:
    # Función dummy para satisfacer la verificación obligatoria de inicio de ZeroGPU
    @spaces.GPU
    def _zerogpu_startup_validator():
        pass

# Inferencia directa en la CPU del contenedor (16 GB RAM disponibles, sin bloqueos de CUDA de ZeroGPU)
def predict_and_explain(image, method, alpha, colormap, target_class_choice):
    return _run_prediction(image, method, alpha, colormap, target_class_choice)

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

class_choices = ["Predicción principal (automática)"] + list(CLASES_DICT.values())

with gr.Blocks(title="AI Chest Disease Detection - RACM") as demo:
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
                target_class_dropdown = gr.Dropdown(
                    choices=class_choices,
                    value="Predicción principal (automática)",
                    label="Clase a explicar en Grad-CAM",
                    info="Predicción principal automática o focalizar una patología específica.",
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
            output_zones = gr.HTML()

        with gr.Column(scale=1):
            gr.Markdown("### 📊 Diagnóstico y Probabilidades")
            output_summary = gr.HTML()
            output_label = gr.Label(num_top_classes=4, label="Distribución de Probabilidad")

    analyze_btn.click(
        fn=predict_and_explain,
        inputs=[input_image, method_radio, alpha_slider, cmap_dropdown, target_class_dropdown],
        outputs=[output_overlay, output_zones, output_label, output_summary],
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
    demo.launch(theme=custom_theme, server_name="0.0.0.0", server_port=7860)
