"""Interfaz web en Streamlit con inferencia multiclase, explicabilidad Grad-CAM++
y diseño visual personalizado para Roberto Alejandro Chagra Martínez (RACM).
"""

from __future__ import annotations

import base64
import json
import logging
import os
import warnings
from pathlib import Path

# Silenciar avisos informativos de TensorFlow/Keras/oneDNN en la consola de Streamlit
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

tf.get_logger().setLevel(logging.ERROR)
try:
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
except Exception:
    pass
try:
    from absl import logging as absl_logging

    absl_logging.set_verbosity(absl_logging.ERROR)
except Exception:
    pass

# En despliegues remotos / cloud (Render, etc.), evitar asignación de memoria GPU inexistente
try:
    tf.config.set_visible_devices([], "GPU")
except Exception:
    pass

try:
    from src.model_utils import (
        IMG_SIZE,
        compute_gradcam_heatmap,
        get_backbone_specs,
        load_model_metadata,
        overlay_gradcam,
        preprocess_pil_image,
        summarize_activation_zones,
    )
except ImportError:
    from model_utils import (
        IMG_SIZE,
        compute_gradcam_heatmap,
        get_backbone_specs,
        load_model_metadata,
        overlay_gradcam,
        preprocess_pil_image,
        summarize_activation_zones,
    )

BASE_DIR = Path(__file__).resolve().parent.parent
LOW_CONFIDENCE_THRESHOLD = 60.0
LOGO_PATH = BASE_DIR / "assets" / "racm_logo.jpg"


def get_base64_image(image_path: Path) -> str | None:
    """Convierte una imagen local a base64 para embeber en HTML sin dependencias de red."""
    if not image_path.exists():
        return None
    try:
        with open(image_path, "rb") as f:
            encoded = base64.b64encode(f.read()).decode("utf-8")
            return f"data:image/jpeg;base64,{encoded}"
    except Exception:
        return None


# Carga del logo para icono de pestaña y avatar
logo_b64 = get_base64_image(LOGO_PATH)
favicon_img = Image.open(LOGO_PATH) if LOGO_PATH.exists() else "🫁"

st.set_page_config(
    page_title="AI Chest Disease Detection • Roberto A. Chagra Martínez (RACM)",
    page_icon=favicon_img,
    layout="wide",
    initial_sidebar_state="expanded",
)

# Inyección de estilos CSS basados en la paleta del logo RACM (Ciberpunk Médico: Azul Neón, Cian, Violeta, Plata y Fondo Espacial Oscuro)
st.markdown(
    """
    <style>
    /* Tipografía y fondo principal */
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'Outfit', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    }

    .stApp {
        background: radial-gradient(circle at 15% 15%, #0d1527 0%, #070b14 55%, #05070d 100%);
        color: #f1f5f9;
    }

    /* Barra lateral */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #090e1a 0%, #0c1424 50%, #080d18 100%) !important;
        border-right: 1px solid rgba(0, 240, 255, 0.15) !important;
        box-shadow: 4px 0 24px rgba(0, 0, 0, 0.4);
    }

    /* Tarjeta de Creador en Sidebar */
    .creator-card {
        background: linear-gradient(135deg, rgba(16, 24, 44, 0.85) 0%, rgba(26, 16, 60, 0.65) 100%);
        border: 1px solid rgba(0, 240, 255, 0.3);
        border-radius: 18px;
        padding: 18px 14px;
        margin-bottom: 22px;
        text-align: center;
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.45), 0 0 16px rgba(0, 240, 255, 0.15);
        position: relative;
        overflow: hidden;
    }

    .creator-card::before {
        content: "";
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 3px;
        background: linear-gradient(90deg, #00f0ff, #a855f7, #00f0ff);
        background-size: 200% auto;
    }

    .creator-avatar-wrap {
        width: 104px;
        height: 104px;
        margin: 0 auto 12px auto;
        border-radius: 50%;
        padding: 3px;
        background: linear-gradient(135deg, #00f0ff 0%, #a855f7 70%, #38bdf8 100%);
        box-shadow: 0 0 20px rgba(0, 240, 255, 0.4), 0 0 35px rgba(168, 85, 247, 0.25);
        display: flex;
        align-items: center;
        justify-content: center;
    }

    .creator-avatar-img {
        width: 100%;
        height: 100%;
        border-radius: 50%;
        object-fit: cover;
        display: block;
        background-color: #0b0f19;
    }

    .creator-name {
        font-size: 1.02rem;
        font-weight: 700;
        letter-spacing: 0.5px;
        color: #ffffff;
        margin: 6px 0 2px 0;
        text-transform: uppercase;
        line-height: 1.25;
        text-shadow: 0 2px 8px rgba(0,0,0,0.6);
    }

    .creator-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: linear-gradient(90deg, rgba(0, 240, 255, 0.15), rgba(168, 85, 247, 0.18));
        border: 1px solid rgba(0, 240, 255, 0.4);
        padding: 3px 10px;
        border-radius: 20px;
        font-size: 0.76rem;
        font-weight: 600;
        color: #00f0ff;
        margin-top: 4px;
        margin-bottom: 6px;
        text-transform: uppercase;
        letter-spacing: 0.8px;
    }

    .pulse-dot {
        width: 7px;
        height: 7px;
        background: #00f0ff;
        border-radius: 50%;
        box-shadow: 0 0 8px #00f0ff;
        display: inline-block;
    }

    .creator-subtitle {
        font-size: 0.78rem;
        color: #94a3b8;
        font-weight: 500;
        letter-spacing: 0.4px;
        margin-bottom: 10px;
    }

    .creator-tags {
        display: flex;
        justify-content: center;
        flex-wrap: wrap;
        gap: 5px;
        margin-top: 6px;
    }

    .creator-tag {
        font-size: 0.68rem;
        font-family: 'JetBrains Mono', monospace;
        background: rgba(15, 23, 42, 0.8);
        border: 1px solid rgba(168, 85, 247, 0.35);
        color: #c084fc;
        padding: 2px 7px;
        border-radius: 6px;
        font-weight: 500;
    }

    /* Hero Banner Principal */
    .hero-container {
        background: linear-gradient(135deg, rgba(13, 21, 39, 0.9) 0%, rgba(20, 14, 45, 0.8) 100%);
        border: 1px solid rgba(0, 240, 255, 0.25);
        border-radius: 20px;
        padding: 26px 30px;
        margin-bottom: 24px;
        position: relative;
        overflow: hidden;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.5), inset 0 0 40px rgba(0, 240, 255, 0.05);
    }

    .hero-container::after {
        content: "";
        position: absolute;
        top: -50px;
        right: -50px;
        width: 200px;
        height: 200px;
        background: radial-gradient(circle, rgba(168, 85, 247, 0.25) 0%, transparent 70%);
        pointer-events: none;
    }

    .hero-pill {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        background: rgba(0, 240, 255, 0.1);
        border: 1px solid rgba(0, 240, 255, 0.3);
        padding: 4px 14px;
        border-radius: 30px;
        font-size: 0.78rem;
        font-weight: 600;
        color: #38bdf8;
        letter-spacing: 0.8px;
        text-transform: uppercase;
        margin-bottom: 12px;
    }

    .hero-title {
        font-size: 2.15rem;
        font-weight: 800;
        letter-spacing: -0.5px;
        background: linear-gradient(90deg, #ffffff 0%, #38bdf8 45%, #c084fc 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 8px;
        line-height: 1.2;
    }

    .hero-desc {
        font-size: 0.98rem;
        color: #94a3b8;
        max-width: 880px;
        line-height: 1.5;
        margin-bottom: 14px;
    }

    .hero-creator-meta {
        display: flex;
        align-items: center;
        flex-wrap: wrap;
        gap: 14px;
        padding-top: 10px;
        border-top: 1px solid rgba(255, 255, 255, 0.08);
        font-size: 0.86rem;
        color: #cbd5e1;
    }

    .hero-creator-meta strong {
        color: #00f0ff;
    }

    .badge-brand {
        background: linear-gradient(135deg, #00f0ff 0%, #a855f7 100%);
        color: #070b14;
        font-weight: 800;
        font-size: 0.72rem;
        padding: 3px 9px;
        border-radius: 6px;
        letter-spacing: 1px;
    }

    /* Tarjetas de Diagnóstico y Métricas */
    .card-box {
        background: rgba(15, 23, 42, 0.75);
        border: 1px solid rgba(0, 240, 255, 0.18);
        border-radius: 14px;
        padding: 16px 18px;
        margin-bottom: 16px;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.3);
        backdrop-filter: blur(8px);
    }

    .result-box-healthy {
        background: linear-gradient(135deg, rgba(6, 78, 59, 0.35) 0%, rgba(15, 23, 42, 0.85) 100%);
        border: 1.5px solid #10b981;
        border-radius: 16px;
        padding: 18px 20px;
        box-shadow: 0 0 20px rgba(16, 185, 129, 0.25);
    }

    .result-box-disease {
        background: linear-gradient(135deg, rgba(127, 29, 29, 0.35) 0%, rgba(88, 28, 135, 0.3) 100%);
        border: 1.5px solid #f43f5e;
        border-radius: 16px;
        padding: 18px 20px;
        box-shadow: 0 0 20px rgba(244, 63, 94, 0.25);
    }

    .result-tag-healthy {
        font-size: 0.78rem;
        font-weight: 700;
        letter-spacing: 1px;
        color: #34d399;
        text-transform: uppercase;
    }

    .result-tag-disease {
        font-size: 0.78rem;
        font-weight: 700;
        letter-spacing: 1px;
        color: #fb7185;
        text-transform: uppercase;
    }

    .result-value {
        font-size: 1.65rem;
        font-weight: 800;
        color: #ffffff;
        margin: 4px 0 6px 0;
    }

    .confidence-meter {
        font-size: 0.95rem;
        color: #cbd5e1;
    }

    .confidence-meter strong {
        color: #00f0ff;
        font-size: 1.12rem;
    }

    /* Barra de progreso personalizada */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #00f0ff 0%, #a855f7 100%) !important;
        border-radius: 10px;
    }

    /* Estilo para File Uploader */
    [data-testid="stFileUploader"] {
        background: rgba(13, 21, 39, 0.65) !important;
        border: 1.5px dashed rgba(0, 240, 255, 0.35) !important;
        border-radius: 16px !important;
        padding: 14px !important;
        transition: all 0.3s ease;
    }

    [data-testid="stFileUploader"]:hover {
        border-color: #00f0ff !important;
        box-shadow: 0 0 18px rgba(0, 240, 255, 0.25) !important;
    }

    /* Botones y controles de formulario */
    .stSelectbox label, .stSlider label, .stRadio label {
        color: #cbd5e1 !important;
        font-weight: 500 !important;
    }

    /* Pie de página con créditos */
    .footer-container {
        margin-top: 45px;
        padding: 22px 28px;
        border-top: 1px solid rgba(0, 240, 255, 0.2);
        background: linear-gradient(180deg, rgba(13, 21, 39, 0.4) 0%, rgba(7, 11, 20, 0.9) 100%);
        border-radius: 16px;
        display: flex;
        justify-content: space-between;
        align-items: center;
        flex-wrap: wrap;
        gap: 14px;
    }

    .footer-brand {
        display: flex;
        align-items: center;
        gap: 12px;
    }

    .footer-logo-thumb {
        width: 44px;
        height: 44px;
        border-radius: 50%;
        border: 1.5px solid #00f0ff;
        box-shadow: 0 0 10px rgba(0, 240, 255, 0.3);
    }

    .footer-text-title {
        font-size: 0.92rem;
        font-weight: 700;
        color: #ffffff;
    }

    .footer-text-sub {
        font-size: 0.78rem;
        color: #94a3b8;
    }

    .footer-chips {
        display: flex;
        gap: 8px;
        flex-wrap: wrap;
    }

    .footer-chip {
        background: rgba(168, 85, 247, 0.15);
        border: 1px solid rgba(168, 85, 247, 0.35);
        color: #d8b4fe;
        font-size: 0.72rem;
        padding: 3px 9px;
        border-radius: 6px;
        font-family: 'JetBrains Mono', monospace;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data
def cargar_diccionario_clases():
    config_path = BASE_DIR / "config" / "mapeo_clases.json"
    if not config_path.exists():
        return None
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


@st.cache_resource
def cargar_modelo_y_meta():
    model_path = BASE_DIR / "models" / "modelo_chest_disease.keras"
    if not model_path.exists():
        return None, None
    try:
        loaded_model = tf.keras.models.load_model(model_path)
    except Exception as exc:
        logging.error(f"Error al cargar models/modelo_chest_disease.keras: {exc}")
        return None, None
    meta = load_model_metadata(BASE_DIR / "config", model=loaded_model)
    return loaded_model, meta


CLASES_DICT = cargar_diccionario_clases()
model, MODEL_META = cargar_modelo_y_meta()

# Renderizado de la Barra Lateral (Sidebar) con Tarjeta de Creador RACM
with st.sidebar:
    # Tarjeta de Creador & Branding de Roberto Alejandro Chagra Martínez
    avatar_html = (
        f'<img src="{logo_b64}" class="creator-avatar-img" alt="Logo RACM" />'
        if logo_b64
        else '<div style="font-size:2rem;line-height:94px;">🫁</div>'
    )

    st.markdown(
        f"""
        <div class="creator-card">
            <div class="creator-avatar-wrap">
                {avatar_html}
            </div>
            <div class="creator-name">Roberto Alejandro<br>Chagra Martínez</div>
            <div class="creator-badge">
                <span class="pulse-dot"></span>
                <span>Creador del Modelo</span>
            </div>
            <div class="creator-subtitle">Artificial Intelligence & Data Science</div>
            <div class="creator-tags">
                <span class="creator-tag">RACM AI</span>
                <span class="creator-tag">Deep Learning</span>
                <span class="creator-tag">Computer Vision</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### ⚙️ Configuración Visual")

    if MODEL_META is not None:
        backbone = MODEL_META.get("backbone", "mobilenet_v2")
        specs = get_backbone_specs()
        backbone_label = specs[backbone].display_name if backbone in specs else backbone
        img_size = tuple(MODEL_META.get("img_size", list(IMG_SIZE)))
        st.caption(f"🧠 **Arquitectura activa:** `{backbone_label}`")
    else:
        backbone = "densenet121"
        img_size = IMG_SIZE

    gradcam_method = st.radio(
        "Método de explicabilidad",
        options=["gradcam++", "gradcam"],
        format_func=lambda m: "Grad-CAM++ (Alta precisión focal)"
        if m == "gradcam++"
        else "Grad-CAM clásico",
    )

    alpha_overlay = st.slider(
        "Opacidad del mapa de calor",
        min_value=0.15,
        max_value=0.75,
        value=0.42,
        step=0.05,
    )

    cmap_choice = st.selectbox(
        "Paleta térmica (Colormap)",
        options=["jet", "inferno", "turbo"],
        index=0,
    )

    if CLASES_DICT is not None:
        class_options = [-1] + list(range(len(CLASES_DICT)))
        target_class_choice = st.selectbox(
            "Clase a explicar en Grad-CAM",
            options=class_options,
            format_func=lambda idx: "Predicción principal (automática)"
            if idx == -1
            else CLASES_DICT[str(idx)],
        )
    else:
        target_class_choice = -1

    st.markdown("---")
    st.markdown(
        """
        <div style="font-size:0.75rem; color:#64748b; text-align:center;">
            Desarrollado y calibrado con TensorFlow & Keras por <strong>Roberto Alejandro Chagra Martínez</strong>
        </div>
        """,
        unsafe_allow_html=True,
    )


# Hero Banner Principal en la Vista Central
st.markdown(
    """
    <div class="hero-container">
        <div class="hero-pill">
            <span class="pulse-dot"></span>
            <span>Sistema Inteligente de Diagnóstico Torácico</span>
        </div>
        <div class="hero-title">Detección de Enfermedades Pulmonares con IA</div>
        <div class="hero-desc">
            Análisis predictivo multiclase sobre radiografías de tórax (CXR) con interpretabilidad biológica en tiempo real mediante mapas de atención <strong>Grad-CAM / Grad-CAM++</strong>.
        </div>
        <div class="hero-creator-meta">
            <span>👤 Creador: <strong>Roberto Alejandro Chagra Martínez</strong></span>
            <span>•</span>
            <span>🔬 Disciplina: <strong>Artificial Intelligence & Data Science</strong></span>
            <span>•</span>
            <span class="badge-brand">RACM</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Verificaciones de artefactos
if CLASES_DICT is None:
    st.error("⚠️ No se encontró `config/mapeo_clases.json`. Ejecuta el entrenamiento primero.")
elif model is None or MODEL_META is None:
    st.error("⚠️ No se encontró `models/modelo_chest_disease.keras`. Ejecuta el entrenamiento primero.")
else:
    uploaded_file = st.file_uploader(
        "📂 Selecciona o arrastra una radiografía de tórax (JPG, JPEG, PNG)...",
        type=["jpg", "jpeg", "png"],
    )

    if uploaded_file is not None:
        image = Image.open(uploaded_file)

        with st.spinner("⚡ Ejecutando inferencia neuronal y calculando gradientes Grad-CAM..."):
            tensor_imagen = preprocess_pil_image(
                image, backbone=backbone, img_size=img_size
            )
            explain_idx = None if target_class_choice == -1 else int(target_class_choice)
            heatmap, explained_idx, predicciones = compute_gradcam_heatmap(
                model=model,
                preprocessed_tensor=tensor_imagen,
                class_index=explain_idx,
                method=gradcam_method,
            )
            overlay_img = overlay_gradcam(
                image, heatmap, alpha=alpha_overlay, cmap_name=cmap_choice
            )
            zone_summary = summarize_activation_zones(heatmap)

            indice_prediccion = int(np.argmax(predicciones))
            clase_detectada = CLASES_DICT[str(indice_prediccion)]
            clase_explicada = CLASES_DICT[str(explained_idx)]
            confianza_detectada = float(predicciones[indice_prediccion] * 100.0)

        col_img1, col_img2, col_res = st.columns([1.0, 1.0, 1.15], gap="medium")

        with col_img1:
            st.markdown("#### 🖼️ Radiografía Original")
            st.image(image, caption="Entrada clínica (CXR)", use_container_width=True)

        with col_img2:
            st.markdown("#### 🎯 Mapa de Atención Grad-CAM")
            st.image(
                overlay_img,
                caption=f"Foco de atención ({gradcam_method.upper()}) para: {clase_explicada}",
                use_container_width=True,
            )

        with col_res:
            st.markdown("#### 📊 Diagnóstico del Modelo")

            es_sano = clase_detectada.lower() in {"healthy", "normal"}
            box_class = "result-box-healthy" if es_sano else "result-box-disease"
            tag_class = "result-tag-healthy" if es_sano else "result-tag-disease"
            tag_text = "ESTADO PULMONAR: NORMAL" if es_sano else "HALLAZGO PATOLÓGICO DETECTADO"

            st.markdown(
                f"""
                <div class="{box_class}">
                    <div class="{tag_class}">{tag_text}</div>
                    <div class="result-value">{clase_detectada}</div>
                    <div class="confidence-meter">
                        Confianza diagnóstica: <strong>{confianza_detectada:.2f}%</strong>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            if confianza_detectada < LOW_CONFIDENCE_THRESHOLD:
                st.warning(
                    f"⚠️ **Confianza reducida (< {LOW_CONFIDENCE_THRESHOLD:.0f}%):** "
                    "El patrón radiológico presenta ambigüedad; se aconseja corroborar con proyección lateral o TAC."
                )

            st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
            st.markdown("**Distribución de probabilidad por patología:**")
            orden = sorted(
                range(len(CLASES_DICT)),
                key=lambda i: float(predicciones[i]),
                reverse=True,
            )
            for i in orden:
                nombre_clase = CLASES_DICT[str(i)]
                porcentaje = float(predicciones[i])
                porcentaje_barra = max(0.0, min(1.0, porcentaje))
                st.write(f"**{nombre_clase}** — `{porcentaje * 100:.2f}%`")
                st.progress(porcentaje_barra)

            if zone_summary["zones"]:
                st.markdown("---")
                st.markdown("**📍 Segmentación de atención pulmonar:**")
                st.caption(
                    f"Región dominante: **{zone_summary['top_zone']}** "
                    f"({zone_summary['top_share']:.1f}% de masa térmica)"
                )
                top_zones = sorted(
                    zone_summary["zones"].items(), key=lambda kv: kv[1], reverse=True
                )[:3]
                for z_name, z_pct in top_zones:
                    st.write(f"- {z_name}: **{z_pct:.1f}%**")

        st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)
        st.info(
            "ℹ️ **Aviso Clínico de Asistencia por IA:** Este modelo ha sido entrenado para fines de investigación y soporte "
            "al diagnóstico. Corrobora siempre en el mapa Grad-CAM++ que los gradientes de activación se ubiquen dentro de "
            "los campos pleuropulmonares y no en marcapasos, electrodos ni rotulación técnica externa."
        )
    else:
        # Estado inicial informativo cuando aún no se sube imagen
        st.markdown(
            """
            <div class="card-box" style="text-align: center; padding: 36px 20px;">
                <div style="font-size: 2.8rem; margin-bottom: 10px;">🩻</div>
                <h3 style="color: #ffffff; margin-bottom: 6px;">Listo para el análisis radiológico</h3>
                <p style="color: #94a3b8; max-width: 620px; margin: 0 auto 16px auto;">
                    Sube una placa de tórax en el recuadro superior para que la red neuronal efectúe
                    la clasificación multiclase y despliegue el mapa térmico de atención Grad-CAM++.
                </p>
                <div style="display: flex; justify-content: center; gap: 10px; flex-wrap: wrap;">
                    <span class="footer-chip">COVID-19</span>
                    <span class="footer-chip">Neumonía Bacteriana</span>
                    <span class="footer-chip">Neumonía Viral</span>
                    <span class="footer-chip">Campos Pulmonares Sanos</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

# Pie de página institucional de Roberto Alejandro Chagra Martínez (RACM)
logo_thumb_html = (
    f'<img src="{logo_b64}" class="footer-logo-thumb" alt="RACM Logo" />'
    if logo_b64
    else ""
)

st.markdown(
    f"""
    <div class="footer-container">
        <div class="footer-brand">
            {logo_thumb_html}
            <div>
                <div class="footer-text-title">Roberto Alejandro Chagra Martínez</div>
                <div class="footer-text-sub">Artificial Intelligence & Data Science • Creador y Desarrollador del Modelo</div>
            </div>
        </div>
        <div class="footer-chips">
            <span class="footer-chip">RACM AI</span>
            <span class="footer-chip">DenseNet-121</span>
            <span class="footer-chip">Grad-CAM++</span>
            <span class="footer-chip">CXR Multiclass</span>
        </div>
        <div style="font-size:0.76rem; color:#64748b;">
            © 2026 RACM • AI in HealthCare
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)
