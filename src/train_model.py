"""Entrenamiento en dos fases (Transfer Learning + Fine-Tuning) con tf.data.

Soporta arquitecturas modernas (`densenet121` por defecto, `convnext_tiny` y
`mobilenet_v2`), lectura optimizada con `tf.data` sobre `data/processed/{train,val}`,
label smoothing para calibración clínica y exportación de metadatos de backbone
(`config/model_config.json`) compatibles con evaluación y Grad-CAM en Streamlit.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "1")

import numpy as np
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras import layers

try:
    from src.model_utils import (
        GRADCAM_LAYER_NAME,
        IMG_SIZE,
        build_classifier_model,
        get_backbone_specs,
        get_preprocess_fn,
        unfreeze_top_layers,
    )
except ImportError:
    from model_utils import (
        GRADCAM_LAYER_NAME,
        IMG_SIZE,
        build_classifier_model,
        get_backbone_specs,
        get_preprocess_fn,
        unfreeze_top_layers,
    )

BASE_DIR = Path(__file__).resolve().parent.parent
AUTOTUNE = tf.data.AUTOTUNE


def build_datasets(
    data_dir: Path,
    backbone: str,
    batch_size: int,
    seed: int,
    use_cache: bool = False,
) -> tuple[tf.data.Dataset, tf.data.Dataset, list[str], dict[int, float]]:
    """Crea pipelines tf.data optimizados y calcula pesos balanceados sin fugas."""
    train_dir = data_dir / "train"
    val_dir = data_dir / "val"

    if not train_dir.exists() or not val_dir.exists():
        raise FileNotFoundError(
            f"No se encontraron '{train_dir}' y '{val_dir}'. "
            "Ejecuta primero:\n"
            "  python scripts/procesar_dataset.py\n"
            "  python scripts/reorganizar_dataset.py"
        )

    raw_train_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        labels="inferred",
        label_mode="categorical",
        color_mode="rgb",
        batch_size=batch_size,
        image_size=IMG_SIZE,
        shuffle=True,
        seed=seed,
    )
    raw_val_ds = tf.keras.utils.image_dataset_from_directory(
        val_dir,
        labels="inferred",
        label_mode="categorical",
        color_mode="rgb",
        batch_size=batch_size,
        image_size=IMG_SIZE,
        shuffle=False,
    )

    class_names = list(raw_train_ds.class_names)
    if list(raw_val_ds.class_names) != class_names:
        raise ValueError(
            f"Desajuste de clases entre train ({class_names}) y val ({raw_val_ds.class_names})."
        )

    # Pesos de clase calculados sobre la distribución real y deduplicada de train
    y_train_indices = []
    print("\nDistribución real en entrenamiento:")
    for idx, cls_name in enumerate(class_names):
        cls_dir = train_dir / cls_name
        count = sum(1 for entry in os.scandir(cls_dir) if entry.is_file())
        y_train_indices.extend([idx] * count)
        print(f" - [{idx}] {cls_name}: {count} imágenes")

    y_array = np.array(y_train_indices, dtype=np.int32)
    pesos_array = compute_class_weight(
        class_weight="balanced", classes=np.unique(y_array), y=y_array
    )
    pesos_dict = {int(i): float(w) for i, w in enumerate(pesos_array)}
    print("\nPesos de clase calculados:")
    for idx, peso in pesos_dict.items():
        print(f" - '{class_names[idx]}': {peso:.2f}x")

    # Augmentation preservando lateralidad anatómica (sin volteo horizontal)
    augmentation = tf.keras.Sequential(
        [
            layers.RandomRotation(factor=7.0 / 360.0, fill_mode="constant", seed=seed),
            layers.RandomZoom(height_factor=0.08, width_factor=0.08, fill_mode="constant", seed=seed),
            layers.RandomTranslation(height_factor=0.05, width_factor=0.05, fill_mode="constant", seed=seed),
            layers.RandomContrast(factor=0.10, seed=seed),
        ],
        name="xray_augmentation",
    )

    preprocess_fn = get_preprocess_fn(backbone)

    def prepare_train(images, labels):
        images = tf.cast(images, tf.float32)
        images = augmentation(images, training=True)
        images = tf.clip_by_value(images, 0.0, 255.0)
        images = preprocess_fn(images)
        return images, labels

    def prepare_val(images, labels):
        images = tf.cast(images, tf.float32)
        images = preprocess_fn(images)
        return images, labels

    if use_cache:
        def pack_uint8(images, labels):
            return tf.cast(tf.round(tf.clip_by_value(images, 0.0, 255.0)), tf.uint8), labels

        raw_train_ds = (
            raw_train_ds.map(pack_uint8, num_parallel_calls=AUTOTUNE)
            .cache()
            .shuffle(buffer_size=1024, seed=seed, reshuffle_each_iteration=True)
        )
        raw_val_ds = raw_val_ds.map(pack_uint8, num_parallel_calls=AUTOTUNE).cache()

    train_ds = raw_train_ds.map(prepare_train, num_parallel_calls=AUTOTUNE).prefetch(AUTOTUNE)
    val_ds = raw_val_ds.map(prepare_val, num_parallel_calls=AUTOTUNE).prefetch(AUTOTUNE)
    return train_ds, val_ds, class_names, pesos_dict


def train(args: argparse.Namespace) -> None:
    print("TensorFlow versión:", tf.__version__)
    specs = get_backbone_specs()
    print(f"Backbone seleccionado: {specs[args.backbone].display_name}")

    tf.keras.utils.set_random_seed(args.seed)

    data_path = Path(args.data_dir)
    data_dir = data_path if data_path.is_absolute() else (BASE_DIR / data_path)
    train_ds, val_ds, class_names, class_weights = build_datasets(
        data_dir,
        backbone=args.backbone,
        batch_size=args.batch_size,
        seed=args.seed,
        use_cache=args.cache,
    )

    # Exportar mapeo dinámico de clases y configuración exacta del modelo
    config_dir = BASE_DIR / "config"
    config_dir.mkdir(parents=True, exist_ok=True)

    indices_clases = {str(idx): name for idx, name in enumerate(class_names)}
    with open(config_dir / "mapeo_clases.json", "w", encoding="utf-8") as f:
        json.dump(indices_clases, f, indent=2, ensure_ascii=False)

    model_config = {
        "backbone": args.backbone,
        "img_size": list(IMG_SIZE),
        "last_conv_layer": GRADCAM_LAYER_NAME,
        "label_smoothing": args.label_smoothing,
        "class_names": class_names,
    }
    with open(config_dir / "model_config.json", "w", encoding="utf-8") as f:
        json.dump(model_config, f, indent=2, ensure_ascii=False)

    print(f"\nConfiguración exportada en 'config/mapeo_clases.json' y 'config/model_config.json'")

    model, base_model = build_classifier_model(
        backbone=args.backbone,
        num_classes=len(class_names),
        img_size=IMG_SIZE,
        dropout_rate=args.dropout,
        l2_reg=args.l2_reg,
    )

    models_dir = BASE_DIR / "models"
    results_dir = BASE_DIR / "results"
    models_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    model_path = models_dir / args.model_name

    loss_fn = tf.keras.losses.CategoricalCrossentropy(
        label_smoothing=args.label_smoothing
    )
    metrics_list = [
        "accuracy",
        tf.keras.metrics.Precision(name="precision"),
        tf.keras.metrics.Recall(name="recall"),
        tf.keras.metrics.AUC(name="auc"),
    ]

    # Fase 1: Entrenar cabezal superior con backbone congelado
    print(f"\n=== FASE 1: Entrenamiento de cabezal ({args.epochs_phase1} épocas) ===")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=args.lr_phase1),
        loss=loss_fn,
        metrics=metrics_list,
    )
    callbacks_p1 = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=3, restore_best_weights=True, verbose=1
        )
    ]
    hist_p1 = model.fit(
        train_ds,
        epochs=args.epochs_phase1,
        validation_data=val_ds,
        class_weight=class_weights,
        callbacks=callbacks_p1,
        shuffle=False,
    )

    # Fase 2: Fine-Tuning del bloque superior manteniendo BatchNorm congelada
    n_unfrozen = unfreeze_top_layers(
        model=model,
        base_model=base_model,
        num_layers=args.unfreeze_layers,
        backbone=args.backbone,
    )
    print(
        f"\n=== FASE 2: Fine-Tuning ({args.epochs_phase2} épocas, "
        f"{n_unfrozen} capas del backbone descongeladas) ==="
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=args.lr_phase2),
        loss=loss_fn,
        metrics=metrics_list,
    )
    callbacks_p2 = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=5, restore_best_weights=True, verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=2, min_lr=1e-7, verbose=1
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(model_path),
            monitor="val_loss",
            save_best_only=True,
            verbose=1,
        ),
    ]
    hist_p2 = model.fit(
        train_ds,
        epochs=args.epochs_phase2,
        validation_data=val_ds,
        class_weight=class_weights,
        callbacks=callbacks_p2,
        shuffle=False,
    )

    if not model_path.exists():
        model.save(model_path)

    history_payload = {
        "backbone": args.backbone,
        "phase1": {k: [float(x) for x in v] for k, v in hist_p1.history.items()},
        "phase2": {k: [float(x) for x in v] for k, v in hist_p2.history.items()},
    }
    with open(results_dir / "training_history.json", "w", encoding="utf-8") as f:
        json.dump(history_payload, f, indent=2)

    print(f"\nEntrenamiento completado. Modelo guardado en '{model_path}'")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Entrena clasificador torácico con tf.data y DenseNet121/ConvNeXt/MobileNetV2."
    )
    parser.add_argument(
        "--backbone",
        choices=["densenet121", "convnext_tiny", "mobilenet_v2"],
        default="densenet121",
        help="Arquitectura base (default: densenet121).",
    )
    parser.add_argument("--data-dir", default="data/processed")
    parser.add_argument("--model-name", default="modelo_chest_disease.keras")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs-phase1", type=int, default=8)
    parser.add_argument("--epochs-phase2", type=int, default=25)
    parser.add_argument("--lr-phase1", type=float, default=1e-3)
    parser.add_argument("--lr-phase2", type=float, default=1e-5)
    parser.add_argument("--unfreeze-layers", type=int, default=None)
    parser.add_argument("--dropout", type=float, default=0.4)
    parser.add_argument("--l2-reg", type=float, default=1e-4)
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    parser.add_argument("--cache", action="store_true", help="Cachear datasets en RAM.")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
