"""Evaluación clínica sobre `data/processed/test` + auditoría visual Grad-CAM.

Detecta automáticamente el backbone activo en `config/model_config.json` (o lo
infiere del modelo cargado), evalúa métricas por clase (Precision, Recall, F1,
ROC-AUC Macro), genera la matriz de confusión y opcionalmente un mosaico con
mapas de calor Grad-CAM sobre muestras de prueba.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "1")

import matplotlib
import numpy as np
import seaborn as sns
import tensorflow as tf
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

try:
    from src.model_utils import (
        IMG_SIZE,
        compute_gradcam_heatmap,
        get_preprocess_fn,
        load_model_metadata,
        overlay_gradcam,
        preprocess_pil_image,
    )
except ImportError:
    from model_utils import (
        IMG_SIZE,
        compute_gradcam_heatmap,
        get_preprocess_fn,
        load_model_metadata,
        overlay_gradcam,
        preprocess_pil_image,
    )

BASE_DIR = Path(__file__).resolve().parent.parent
AUTOTUNE = tf.data.AUTOTUNE


def save_gradcam_audit_grid(
    model: tf.keras.Model,
    backbone: str,
    file_paths: list[str],
    y_true: np.ndarray,
    y_pred: np.ndarray,
    probs: np.ndarray,
    class_names: list[str],
    output_path: Path,
    max_samples: int = 8,
) -> None:
    """Genera un mosaico Grad-CAM++ con aciertos y errores reales del test set."""
    import matplotlib.pyplot as plt

    errors_idx = np.where(y_true != y_pred)[0].tolist()
    correct_idx = np.where(y_true == y_pred)[0].tolist()

    # Seleccionar 1 acierto por cada clase (si existe) + errores variados entre clases
    chosen: list[int] = []
    for cls_i in range(len(class_names)):
        cls_hits = [idx for idx in correct_idx if int(y_true[idx]) == cls_i]
        if cls_hits and len(chosen) < max_samples:
            chosen.append(cls_hits[len(cls_hits) // 2])

    n_err_slots = min(len(errors_idx), max(0, max_samples - len(chosen)))
    if n_err_slots > 0:
        err_step = max(1, len(errors_idx) // n_err_slots)
        chosen.extend(errors_idx[::err_step][:n_err_slots])

    if len(chosen) < max_samples:
        remaining = [idx for idx in correct_idx if idx not in chosen]
        chosen.extend(remaining[: max_samples - len(chosen)])
    chosen = chosen[:max_samples]

    if not chosen:
        return

    cols = min(4, len(chosen))
    rows = int(np.ceil(len(chosen) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4.5 * cols, 4.5 * rows))
    axes_flat = np.atleast_1d(axes).ravel()

    for ax_i, sample_idx in enumerate(chosen):
        img_pil = Image.open(file_paths[sample_idx]).convert("RGB")
        tensor = preprocess_pil_image(img_pil, backbone=backbone, img_size=IMG_SIZE)
        heatmap, _, _ = compute_gradcam_heatmap(
            model, tensor, class_index=int(y_pred[sample_idx]), method="gradcam++"
        )
        overlay = overlay_gradcam(img_pil, heatmap, alpha=0.42)

        true_lbl = class_names[int(y_true[sample_idx])]
        pred_lbl = class_names[int(y_pred[sample_idx])]
        conf = float(probs[sample_idx, int(y_pred[sample_idx])] * 100.0)
        ok = y_true[sample_idx] == y_pred[sample_idx]
        title_color = "#117a37" if ok else "#b00020"

        axes_flat[ax_i].imshow(overlay)
        axes_flat[ax_i].set_title(
            f"Real: {true_lbl}\nPred: {pred_lbl} ({conf:.1f}%)",
            fontsize=9.5,
            color=title_color,
            fontweight="bold",
        )
        axes_flat[ax_i].axis("off")

    for j in range(len(chosen), len(axes_flat)):
        axes_flat[j].axis("off")

    plt.suptitle(f"Auditoría Visual Grad-CAM++ ({backbone})", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=220)
    plt.close(fig)


def evaluate(args: argparse.Namespace) -> None:
    if not args.show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    config_dir = BASE_DIR / "config"
    config_path = config_dir / "mapeo_clases.json"
    model_path = BASE_DIR / "models" / args.model_name
    test_path = Path(args.test_dir)
    test_dir = test_path if test_path.is_absolute() else (BASE_DIR / test_path)

    if not config_path.exists():
        raise FileNotFoundError(f"No se encontró '{config_path}'.")
    if not model_path.exists():
        raise FileNotFoundError(f"No se encontró '{model_path}'.")
    if not test_dir.exists():
        raise FileNotFoundError(
            f"No se encontró el directorio de prueba '{test_dir}'. "
            "Ejecuta scripts/procesar_dataset.py y scripts/reorganizar_dataset.py."
        )

    with open(config_path, "r", encoding="utf-8") as f:
        diccionario_clases = json.load(f)
    nombres_clases = [diccionario_clases[str(i)] for i in range(len(diccionario_clases))]

    print(f"Cargando modelo desde '{model_path}'...")
    model = tf.keras.models.load_model(model_path)
    meta = load_model_metadata(config_dir, model=model)
    backbone = args.backbone or meta.get("backbone", "mobilenet_v2")
    preprocess_fn = get_preprocess_fn(backbone)
    print(f"Backbone detectado para evaluación: {backbone}")

    test_ds = tf.keras.utils.image_dataset_from_directory(
        test_dir,
        labels="inferred",
        label_mode="categorical",
        color_mode="rgb",
        batch_size=args.batch_size,
        image_size=IMG_SIZE,
        shuffle=False,
    )

    if list(test_ds.class_names) != nombres_clases:
        raise ValueError(
            f"Las clases en '{test_dir}' ({test_ds.class_names}) no coinciden con "
            f"mapeo_clases.json ({nombres_clases})."
        )

    file_paths = list(test_ds.file_paths)
    y_true_onehot = np.concatenate([y.numpy() for _, y in test_ds], axis=0)
    etiquetas_reales = np.argmax(y_true_onehot, axis=1)

    prepared_ds = test_ds.map(
        lambda x, y: (preprocess_fn(tf.cast(x, tf.float32)), y),
        num_parallel_calls=AUTOTUNE,
    ).prefetch(AUTOTUNE)

    print(f"\nEvaluando {len(etiquetas_reales)} imágenes en '{test_dir}'...")
    predicciones_crudas = model.predict(prepared_ds, verbose=1)
    predicciones_clases = np.argmax(predicciones_crudas, axis=1)

    print("\n" + "=" * 60)
    print("REPORTE DE CLASIFICACIÓN CLÍNICA")
    print("=" * 60)
    if len(etiquetas_reales) < 100:
        print(
            "ADVERTENCIA: Conjunto de prueba pequeño (<100 imágenes); "
            "los intervalos de confianza pueden ser amplios.\n"
        )

    report_txt = classification_report(
        etiquetas_reales, predicciones_clases, target_names=nombres_clases, digits=4
    )
    report_dict = classification_report(
        etiquetas_reales,
        predicciones_clases,
        target_names=nombres_clases,
        output_dict=True,
    )
    print(report_txt)

    macro_auc = None
    try:
        macro_auc = float(
            roc_auc_score(
                y_true_onehot, predicciones_crudas, multi_class="ovr", average="macro"
            )
        )
        print(f"ROC-AUC Macro (OvR): {macro_auc:.4f}")
    except ValueError as exc:
        print(f"ROC-AUC no calculable: {exc}")

    matriz = confusion_matrix(etiquetas_reales, predicciones_clases)
    row_sums = matriz.sum(axis=1, keepdims=True)
    matriz_pct = (
        np.divide(
            matriz,
            row_sums,
            out=np.zeros_like(matriz, dtype=float),
            where=row_sums != 0,
        )
        * 100.0
    )
    annot_labels = np.array(
        [
            [f"{matriz[i, j]}\n({matriz_pct[i, j]:.1f}%)" for j in range(matriz.shape[1])]
            for i in range(matriz.shape[0])
        ]
    )

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        matriz_pct,
        annot=annot_labels,
        fmt="",
        cmap="Blues",
        vmin=0,
        vmax=100,
        xticklabels=nombres_clases,
        yticklabels=nombres_clases,
    )
    plt.title(
        f"Matriz de Confusión ({backbone}) - Conteo y Recall %", pad=20, fontsize=14
    )
    plt.ylabel("Diagnóstico Real", fontsize=12, fontweight="bold")
    plt.xlabel("Predicción del Modelo", fontsize=12, fontweight="bold")
    plt.xticks(rotation=30, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()

    results_dir = BASE_DIR / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    cm_path = results_dir / "matriz_confusion.png"
    plt.savefig(cm_path, dpi=300)

    try:
        test_dir_label = str(test_dir.resolve().relative_to(BASE_DIR.resolve()))
    except ValueError:
        test_dir_label = str(test_dir.resolve())

    metrics_payload = {
        "backbone": backbone,
        "test_dir": test_dir_label,
        "num_samples": int(len(etiquetas_reales)),
        "roc_auc_macro_ovr": macro_auc,
        "classification_report": report_dict,
        "confusion_matrix_counts": matriz.tolist(),
        "class_names": nombres_clases,
    }
    metrics_path = results_dir / "metricas_evaluacion.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics_payload, f, indent=2, ensure_ascii=False)

    print(f"\nMatriz guardada en: {cm_path}")
    print(f"Métricas guardadas en: {metrics_path}")

    if args.gradcam_samples > 0:
        gc_path = results_dir / "gradcam_auditoria.png"
        save_gradcam_audit_grid(
            model=model,
            backbone=backbone,
            file_paths=file_paths,
            y_true=etiquetas_reales,
            y_pred=predicciones_clases,
            probs=predicciones_crudas,
            class_names=nombres_clases,
            output_path=gc_path,
            max_samples=args.gradcam_samples,
        )
        print(f"Mosaico Grad-CAM++ guardado en: {gc_path}")

    if args.show:
        plt.show()
    else:
        plt.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evalúa el modelo entrenado sobre test.")
    parser.add_argument("--test-dir", default="data/processed/test")
    parser.add_argument("--model-name", default="modelo_chest_disease.keras")
    parser.add_argument(
        "--backbone",
        choices=["densenet121", "convnext_tiny", "mobilenet_v2"],
        default=None,
        help="Sobrescribir detección automática de backbone.",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--gradcam-samples",
        type=int,
        default=8,
        help="Número de placas a exportar con Grad-CAM++ (0 para desactivar).",
    )
    parser.add_argument("--show", action="store_true", help="Mostrar gráfico en ventana.")
    return parser.parse_args()


if __name__ == "__main__":
    evaluate(parse_args())
