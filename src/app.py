"""Interfaz web en Streamlit con inferencia multiclase y explicabilidad Grad-CAM."""

from __future__ import annotations

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

st.set_page_config(
    page_title="AI Chest Disease Detection + Grad-CAM",
    page_icon="🫁",
    layout="wide",
)

st.markdown(
    """
    <style>
    .main { background-color: #fafafa; }
    .stAlert { border-radius: 10px; }
    h1 { color: #1e3d59; }
    h3 { color: #17b978; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🫁 Diagnóstico Automatizado de Enfermedades Torácicas")
st.write(
    "Carga una radiografía de tórax para analizar patrones patológicos y visualizar "
    "en qué regiones pulmonares se concentra la red neuronal (**Grad-CAM / Grad-CAM++**)."
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
    except Exception:
        return None, None
    meta = load_model_metadata(BASE_DIR / "config", model=loaded_model)
    return loaded_model, meta


CLASES_DICT = cargar_diccionario_clases()
model, MODEL_META = cargar_modelo_y_meta()

if CLASES_DICT is None:
    st.error("⚠️ No se encontró `config/mapeo_clases.json`. Ejecuta el entrenamiento primero.")
elif model is None or MODEL_META is None:
    st.error("⚠️ No se encontró `models/modelo_chest_disease.keras`. Ejecuta el entrenamiento primero.")
else:
    backbone = MODEL_META.get("backbone", "mobilenet_v2")
    specs = get_backbone_specs()
    backbone_label = specs[backbone].display_name if backbone in specs else backbone
    img_size = tuple(MODEL_META.get("img_size", list(IMG_SIZE)))

    with st.sidebar:
        st.header("Configuración Visual")
        st.caption(f"**Arquitectura activa:** {backbone_label}")
        gradcam_method = st.radio(
            "Método de explicabilidad",
            options=["gradcam++", "gradcam"],
            format_func=lambda m: "Grad-CAM++ (Más preciso en focos múltiples)"
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
        cmap_choice = st.selectbox("Paleta térmica", options=["jet", "inferno", "turbo"], index=0)

        class_options = [-1] + list(range(len(CLASES_DICT)))
        target_class_choice = st.selectbox(
            "Clase a explicar en Grad-CAM",
            options=class_options,
            format_func=lambda idx: "Predicción principal (automática)"
            if idx == -1
            else CLASES_DICT[str(idx)],
        )

    uploaded_file = st.file_uploader(
        "Selecciona una radiografía en formato JPG, JPEG o PNG...",
        type=["jpg", "jpeg", "png"],
    )

    if uploaded_file is not None:
        image = Image.open(uploaded_file)

        with st.spinner("Ejecutando inferencia y generando mapa Grad-CAM..."):
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

        col_img1, col_img2, col_res = st.columns([1.0, 1.0, 1.1])

        with col_img1:
            st.subheader("Placa Original")
            st.image(image, caption="Radiografía de entrada", use_container_width=True)

        with col_img2:
            st.subheader("Mapa de Calor Grad-CAM")
            st.image(
                overlay_img,
                caption=f"Foco de atención para: {clase_explicada}",
                use_container_width=True,
            )

        with col_res:
            st.subheader("Análisis de Red Neuronal")
            if clase_detectada.lower() in {"healthy", "normal"}:
                st.success(f"### Resultado: {clase_detectada}")
                st.write(f"Confianza del análisis: **{confianza_detectada:.2f}%**")
            else:
                st.error(f"### Hallazgo: {clase_detectada}")
                st.write(f"Confianza del análisis: **{confianza_detectada:.2f}%**")

            if confianza_detectada < LOW_CONFIDENCE_THRESHOLD:
                st.warning(
                    f"⚠️ Confianza baja (< {LOW_CONFIDENCE_THRESHOLD:.0f}%): "
                    "resultado incierto; requiere revisión radiológica prioritaria."
                )

            st.write("**Distribución de confianza por patología:**")
            orden = sorted(
                range(len(CLASES_DICT)),
                key=lambda i: float(predicciones[i]),
                reverse=True,
            )
            for i in orden:
                nombre_clase = CLASES_DICT[str(i)]
                porcentaje = float(predicciones[i])
                porcentaje_barra = max(0.0, min(1.0, porcentaje))
                st.write(f"{nombre_clase} ({porcentaje * 100:.1f}%)")
                st.progress(porcentaje_barra)

            if zone_summary["zones"]:
                st.write("---")
                st.write("**Zonas de mayor activación (convención radiológica):**")
                st.caption(
                    f"Foco principal: **{zone_summary['top_zone']}** "
                    f"({zone_summary['top_share']:.1f}% de activación)"
                )
                top_zones = sorted(
                    zone_summary["zones"].items(), key=lambda kv: kv[1], reverse=True
                )[:3]
                for z_name, z_pct in top_zones:
                    st.write(f"- {z_name}: **{z_pct:.1f}%**")

        st.info(
            "ℹ️ *Aviso clínico: Este sistema de IA es apoyo experimental para docencia e "
            "investigación; verifica en el mapa Grad-CAM que la red enfoque los campos "
            "pulmonares y no artefactos o etiquetas externas.*"
        )
