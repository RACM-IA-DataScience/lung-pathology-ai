"""Utilidades compartidas de arquitectura, preprocesamiento y Grad-CAM.

Centraliza la construcción de backbones (DenseNet121, ConvNeXtTiny, MobileNetV2),
su preprocesamiento específico y la generación de mapas de calor Grad-CAM /
Grad-CAM++ compatibles tanto con los modelos Functional nuevos como con modelos
Sequential previos.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Optional, Tuple

import matplotlib
import numpy as np
import tensorflow as tf
from PIL import Image
from tensorflow.keras import layers, models

IMG_SIZE = (224, 224)
GRADCAM_LAYER_NAME = "gradcam_target"


@dataclass(frozen=True)
class BackboneSpec:
    name: str
    display_name: str
    builder: Callable[..., tf.keras.Model]
    preprocess_fn: Callable[[tf.Tensor | np.ndarray], tf.Tensor | np.ndarray]
    default_unfreeze_layers: int
    head_units: int


def get_backbone_specs() -> Dict[str, BackboneSpec]:
    return {
        "densenet121": BackboneSpec(
            name="densenet121",
            display_name="DenseNet-121 (Estándar Clínico CXR)",
            builder=tf.keras.applications.DenseNet121,
            preprocess_fn=tf.keras.applications.densenet.preprocess_input,
            default_unfreeze_layers=60,
            head_units=256,
        ),
        "convnext_tiny": BackboneSpec(
            name="convnext_tiny",
            display_name="ConvNeXt-Tiny (Arquitectura Moderna)",
            builder=tf.keras.applications.ConvNeXtTiny,
            preprocess_fn=tf.keras.applications.convnext.preprocess_input,
            default_unfreeze_layers=40,
            head_units=256,
        ),
        "mobilenet_v2": BackboneSpec(
            name="mobilenet_v2",
            display_name="MobileNetV2 (Ligero)",
            builder=tf.keras.applications.MobileNetV2,
            preprocess_fn=tf.keras.applications.mobilenet_v2.preprocess_input,
            default_unfreeze_layers=20,
            head_units=128,
        ),
    }


def get_preprocess_fn(backbone: str) -> Callable:
    specs = get_backbone_specs()
    if backbone not in specs:
        raise ValueError(
            f"Backbone '{backbone}' no soportado. Opciones: {list(specs.keys())}"
        )
    return specs[backbone].preprocess_fn


def build_classifier_model(
    backbone: str,
    num_classes: int,
    img_size: Tuple[int, int] = IMG_SIZE,
    dropout_rate: float = 0.4,
    l2_reg: float = 1e-4,
) -> Tuple[tf.keras.Model, tf.keras.Model]:
    """Construye un modelo Functional que expone directamente `gradcam_target`."""
    specs = get_backbone_specs()
    if backbone not in specs:
        raise ValueError(
            f"Backbone '{backbone}' no soportado. Opciones: {list(specs.keys())}"
        )
    spec = specs[backbone]

    base_model = spec.builder(
        weights="imagenet",
        include_top=False,
        input_shape=(*img_size, 3),
    )
    base_model.trainable = False

    # Exponer explícitamente la salida espacial 4D antes del GlobalAveragePooling2D
    # para que Grad-CAM funcione igual en DenseNet121, ConvNeXtTiny y MobileNetV2.
    spatial_features = layers.Activation("linear", name=GRADCAM_LAYER_NAME)(
        base_model.output
    )
    x = layers.GlobalAveragePooling2D(name="global_avg_pool")(spatial_features)
    x = layers.Dense(
        spec.head_units,
        activation="relu",
        kernel_regularizer=tf.keras.regularizers.l2(l2_reg),
        name="classifier_dense",
    )(x)
    x = layers.Dropout(dropout_rate, name="classifier_dropout")(x)
    logits = layers.Dense(num_classes, name="logits")(x)
    outputs = layers.Activation("softmax", name="predictions")(logits)

    model = models.Model(inputs=base_model.input, outputs=outputs, name=f"cxr_{backbone}")
    return model, base_model


def unfreeze_top_layers(
    model: tf.keras.Model,
    base_model: tf.keras.Model,
    num_layers: Optional[int] = None,
    backbone: str = "densenet121",
) -> int:
    """Descongela las últimas N capas del backbone manteniendo BatchNorm congelada."""
    spec = get_backbone_specs()[backbone]
    n_unfreeze = num_layers if num_layers is not None else spec.default_unfreeze_layers

    # Como en Functional API las capas de base_model forman parte directa de `model`,
    # descongelamos sobre base_model.layers y protegemos cualquier BatchNormalization.
    base_model.trainable = True
    cutoff = max(0, len(base_model.layers) - n_unfreeze)
    trainable_count = 0

    for idx, layer in enumerate(base_model.layers):
        if idx < cutoff or isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False
        else:
            layer.trainable = True
            trainable_count += 1

    # Asegurar que el cabezal permanezca entrenable
    for layer in model.layers:
        if layer.name in {"classifier_dense", "classifier_dropout", "logits", "predictions"}:
            layer.trainable = True

    return trainable_count


def infer_backbone_from_model(model: tf.keras.Model) -> str:
    """Infiere el backbone cuando no existe `config/model_config.json`."""
    layer_names = " ".join(layer.name.lower() for layer in model.layers)
    if "convnext" in layer_names:
        return "convnext_tiny"
    if "densenet" in layer_names or "conv5_block16" in layer_names:
        return "densenet121"
    return "mobilenet_v2"


def load_model_metadata(config_dir: Path, model: Optional[tf.keras.Model] = None) -> dict:
    """Carga `config/model_config.json` o construye metadatos compatibles."""
    model_cfg_path = config_dir / "model_config.json"
    if model_cfg_path.exists():
        with open(model_cfg_path, "r", encoding="utf-8") as f:
            return json.load(f)

    backbone = infer_backbone_from_model(model) if model is not None else "mobilenet_v2"
    return {
        "backbone": backbone,
        "img_size": list(IMG_SIZE),
        "last_conv_layer": GRADCAM_LAYER_NAME,
    }


def preprocess_pil_image(
    imagen_pil: Image.Image,
    backbone: str,
    img_size: Tuple[int, int] = IMG_SIZE,
) -> np.ndarray:
    """Preprocesa una imagen PIL de forma idéntica al pipeline de entrenamiento."""
    img = imagen_pil.convert("RGB").resize(img_size, Image.Resampling.LANCZOS)
    img_array = np.asarray(img, dtype=np.float32)
    img_batch = np.expand_dims(img_array, axis=0)
    preprocess_fn = get_preprocess_fn(backbone)
    return np.asarray(preprocess_fn(img_batch), dtype=np.float32)


def _find_last_4d_layer(container: tf.keras.Model) -> tf.keras.layers.Layer:
    for layer in reversed(container.layers):
        try:
            shape = layer.output.shape
            if len(shape) == 4 and shape[1] is not None and shape[1] > 1:
                return layer
        except Exception:
            continue
    raise ValueError("No se encontró una capa espacial 4D para calcular Grad-CAM.")


def _extract_pre_softmax_scores(
    penultimate: tf.Tensor, final_layer: tf.keras.layers.Layer
) -> tf.Tensor:
    """Obtiene logits pre-softmax para evitar gradientes nulos cuando softmax=1.0."""
    if isinstance(final_layer, tf.keras.layers.Dense) and len(final_layer.weights) >= 1:
        kernel = final_layer.kernel
        logits = tf.matmul(penultimate, kernel)
        if final_layer.use_bias:
            logits = tf.nn.bias_add(logits, final_layer.bias)
        return logits
    return final_layer(penultimate, training=False)


def compute_gradcam_heatmap(
    model: tf.keras.Model,
    preprocessed_tensor: np.ndarray,
    class_index: Optional[int] = None,
    method: str = "gradcam++",
) -> Tuple[np.ndarray, int, np.ndarray]:
    """Calcula mapa de calor Grad-CAM o Grad-CAM++ usando logits pre-softmax.

    Retorna `(heatmap_2d_norm, predicted_or_target_idx, predictions_1d)`.
    """
    x_tensor = tf.convert_to_tensor(preprocessed_tensor, dtype=tf.float32)

    layer_names = {l.name for l in model.layers}
    use_direct_graph = GRADCAM_LAYER_NAME in layer_names

    if use_direct_graph:
        target_layer = model.get_layer(GRADCAM_LAYER_NAME)
        if "logits" in layer_names:
            score_tensor = model.get_layer("logits").output
            grad_model = tf.keras.Model(
                inputs=model.inputs,
                outputs=[target_layer.output, score_tensor, model.output],
            )
            with tf.GradientTape() as tape:
                conv_outputs, logits, preds = grad_model(x_tensor, training=False)
                if isinstance(preds, list):
                    preds = preds[0]
                target_idx = int(tf.argmax(preds[0])) if class_index is None else int(class_index)
                score = logits[:, target_idx]
            grads = tape.gradient(score, conv_outputs)
        else:
            grad_model = tf.keras.Model(
                inputs=model.inputs,
                outputs=[target_layer.output, model.output],
            )
            with tf.GradientTape() as tape:
                conv_outputs, preds = grad_model(x_tensor, training=False)
                if isinstance(preds, list):
                    preds = preds[0]
                target_idx = int(tf.argmax(preds[0])) if class_index is None else int(class_index)
                # Usar log-probabilidades si no hay capa de logits explícita
                score = tf.math.log(tf.clip_by_value(preds[:, target_idx], 1e-7, 1.0))
            grads = tape.gradient(score, conv_outputs)
    else:
        # Compatibilidad con Sequential antiguo que envuelve un sub-modelo base
        sub_model_idx = None
        for idx, layer in enumerate(model.layers):
            if isinstance(layer, tf.keras.Model):
                sub_model_idx = idx
                break

        if sub_model_idx is None:
            target_layer = _find_last_4d_layer(model)
            grad_model = tf.keras.Model(
                inputs=model.inputs,
                outputs=[target_layer.output, model.output],
            )
            with tf.GradientTape() as tape:
                conv_outputs, preds = grad_model(x_tensor, training=False)
                if isinstance(preds, list):
                    preds = preds[0]
                target_idx = int(tf.argmax(preds[0])) if class_index is None else int(class_index)
                score = tf.math.log(tf.clip_by_value(preds[:, target_idx], 1e-7, 1.0))
            grads = tape.gradient(score, conv_outputs)
        else:
            base_submodel = model.layers[sub_model_idx]
            head_layers = model.layers[sub_model_idx + 1 :]

            if len(base_submodel.output.shape) == 4:
                with tf.GradientTape() as tape:
                    h = x_tensor
                    for pre_layer in model.layers[:sub_model_idx]:
                        h = pre_layer(h, training=False)
                    conv_outputs = base_submodel(h, training=False)
                    if isinstance(conv_outputs, list):
                        conv_outputs = conv_outputs[0]
                    tape.watch(conv_outputs)
                    y = conv_outputs
                    for head_layer in head_layers[:-1]:
                        y = head_layer(y, training=False)
                    final_layer = head_layers[-1]
                    logits = _extract_pre_softmax_scores(y, final_layer)
                    preds = final_layer(y, training=False)
                    target_idx = int(tf.argmax(preds[0])) if class_index is None else int(class_index)
                    score = logits[:, target_idx]
                grads = tape.gradient(score, conv_outputs)
            else:
                last_conv = _find_last_4d_layer(base_submodel)
                grad_model = tf.keras.Model(
                    inputs=base_submodel.inputs,
                    outputs=[last_conv.output, base_submodel.output],
                )
                with tf.GradientTape() as tape:
                    h = x_tensor
                    for pre_layer in model.layers[:sub_model_idx]:
                        h = pre_layer(h, training=False)
                    conv_outputs, base_out = grad_model(h, training=False)
                    tape.watch(conv_outputs)
                    y = base_out
                    for head_layer in head_layers[:-1]:
                        y = head_layer(y, training=False)
                    final_layer = head_layers[-1]
                    logits = _extract_pre_softmax_scores(y, final_layer)
                    preds = final_layer(y, training=False)
                    target_idx = int(tf.argmax(preds[0])) if class_index is None else int(class_index)
                    score = logits[:, target_idx]
                grads = tape.gradient(score, conv_outputs)

    preds_np = preds[0].numpy()
    if grads is None:
        h, w = int(conv_outputs.shape[1]), int(conv_outputs.shape[2])
        return np.zeros((h, w), dtype=np.float32), target_idx, preds_np

    conv_val = conv_outputs[0]
    grads_val = grads[0]

    if method == "gradcam++":
        grads_sq = tf.square(grads_val)
        grads_cu = grads_sq * grads_val
        sum_conv_grads_cu = tf.reduce_sum(conv_val * grads_cu, axis=(0, 1), keepdims=True)
        denom = 2.0 * grads_sq + sum_conv_grads_cu + 1e-8
        alphas = grads_sq / denom
        weights = tf.reduce_sum(alphas * tf.nn.relu(grads_val), axis=(0, 1))
    else:
        weights = tf.reduce_mean(grads_val, axis=(0, 1))

    cam = tf.reduce_sum(conv_val * weights, axis=-1)
    cam = tf.nn.relu(cam).numpy().astype(np.float32)

    max_val = float(np.max(cam))
    if max_val > 1e-8:
        cam = cam / max_val
    else:
        cam = np.zeros_like(cam, dtype=np.float32)

    return cam, target_idx, preds_np


def overlay_gradcam(
    imagen_pil: Image.Image,
    heatmap: np.ndarray,
    alpha: float = 0.42,
    cmap_name: str = "jet",
) -> Image.Image:
    """Superpone el heatmap Grad-CAM sobre la radiografía original usando PIL + Matplotlib."""
    base_rgb = imagen_pil.convert("RGB")
    w, h = base_rgb.size

    heatmap_uint8 = np.uint8(np.clip(heatmap, 0.0, 1.0) * 255)
    heatmap_resized = Image.fromarray(heatmap_uint8).resize(
        (w, h), resample=Image.Resampling.BICUBIC
    )
    heatmap_norm = np.asarray(heatmap_resized, dtype=np.float32) / 255.0

    cmap = matplotlib.colormaps.get_cmap(cmap_name)
    colored_rgba = cmap(heatmap_norm)
    colored_rgb = np.uint8(colored_rgba[:, :, :3] * 255.0)

    base_arr = np.asarray(base_rgb, dtype=np.float32)
    blended = (1.0 - alpha) * base_arr + alpha * colored_rgb.astype(np.float32)
    blended = np.clip(blended, 0.0, 255.0).astype(np.uint8)
    return Image.fromarray(blended)


def summarize_activation_zones(heatmap: np.ndarray) -> dict:
    """Estima la distribución de activación Grad-CAM en 6 zonas radiológicas.

    Nota anatómica estándar (radiografía frontal PA/AP):
    - Lado izquierdo de la imagen (`x < 0.5`) = Hemitórax Derecho del paciente.
    - Lado derecho de la imagen (`x >= 0.5`) = Hemitórax Izquierdo del paciente.
    """
    h, w = heatmap.shape
    total_energy = float(np.sum(heatmap))
    if total_energy <= 1e-8:
        return {
            "top_zone": "Sin focalización clara",
            "top_share": 0.0,
            "zones": {},
        }

    row_splits = [(0, h // 3, "Superior / Apical"), (h // 3, (2 * h) // 3, "Media / Perihiliar"), ((2 * h) // 3, h, "Inferior / Basal")]
    col_splits = [(0, w // 2, "Pulmón Derecho"), (w // 2, w, "Pulmón Izquierdo")]

    zones: Dict[str, float] = {}
    for y0, y1, v_label in row_splits:
        for x0, x1, h_label in col_splits:
            patch = heatmap[y0:y1, x0:x1]
            share = float(np.sum(patch) / total_energy) * 100.0
            zones[f"{h_label} - Zona {v_label}"] = round(share, 1)

    top_zone, top_share = max(zones.items(), key=lambda kv: kv[1])
    return {
        "top_zone": top_zone,
        "top_share": top_share,
        "zones": zones,
    }
