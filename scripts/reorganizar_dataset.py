"""Particionamiento estratificado y agrupado por paciente (cero data leakage).

Divide las imágenes limpias de `data/clean/` (o re-particiona `data/processed/`)
en `data/processed/train/`, `data/processed/val/` y `data/processed/test/`
garantizando mediante `GroupShuffleSplit` que todas las radiografías de un mismo
paciente queden en un único split y que no existan hashes duplicados entre splits.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import os
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image
from sklearn.model_selection import GroupShuffleSplit

try:
    from scripts.procesar_dataset import extract_patient_id
except ImportError:
    from procesar_dataset import extract_patient_id

BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_CLEAN_DIR = "data/clean"
DEFAULT_PROCESSED_DIR = "data/processed"
DEFAULT_CLEAN_MANIFEST = "data/clean_manifest.csv"
DEFAULT_DATASET_MANIFEST = "data/dataset_manifest.csv"
SPLITS = ("train", "val", "test")
VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def resolve_path(path_value: str | Path, base: Path = BASE_DIR) -> Path:
    p = Path(path_value)
    return p if p.is_absolute() else (base / p)


def safe_relpath(path: Path, base: Path = BASE_DIR) -> str:
    try:
        return str(path.resolve().relative_to(base.resolve()))
    except ValueError:
        return str(path.resolve())


def load_clean_manifest(manifest_path: Path) -> Dict[Tuple[str, str], dict]:
    """Carga metadatos previos (patient_id, sha256) si existe `clean_manifest.csv`."""
    lookup: Dict[Tuple[str, str], dict] = {}
    if not manifest_path.exists():
        return lookup
    with open(manifest_path, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            lookup[(row["class"], row["filename"])] = row
    return lookup


def compute_rgb224_hash(path: Path) -> str:
    """Calcula SHA-256 sobre los píxeles RGB 224x224 de la imagen."""
    with Image.open(path) as img:
        norm = img.convert("RGB")
        if norm.size != (224, 224):
            norm = norm.resize((224, 224), Image.Resampling.LANCZOS)
        return hashlib.sha256(norm.tobytes()).hexdigest()


def collect_items(
    clean_dir: Path,
    processed_dir: Path,
    clean_manifest_lookup: Dict[Tuple[str, str], dict],
) -> Tuple[List[dict], str]:
    """Recolecta imágenes limpias desde `data/clean` o desde `data/processed`."""
    items: List[dict] = []

    if clean_dir.exists() and any(p.is_dir() for p in clean_dir.iterdir()):
        source_mode = "clean"
        roots = [clean_dir]
    elif processed_dir.exists() and any((processed_dir / s).exists() for s in SPLITS):
        source_mode = "processed"
        roots = [processed_dir / s for s in SPLITS if (processed_dir / s).exists()]
    else:
        raise FileNotFoundError(
            f"No se encontraron imágenes en '{clean_dir}' ni en '{processed_dir}'. "
            "Ejecuta primero: python scripts/procesar_dataset.py"
        )

    print(f"Recolectando imágenes desde modo '{source_mode}'...")
    for root in roots:
        for class_dir in sorted(root.iterdir()):
            if not class_dir.is_dir() or class_dir.name.startswith((".", "_")):
                continue
            class_name = class_dir.name
            with os.scandir(class_dir) as entries:
                for entry in entries:
                    if not entry.is_file():
                        continue
                    ext = os.path.splitext(entry.name)[1].lower()
                    if ext not in VALID_EXTENSIONS:
                        continue
                    meta = clean_manifest_lookup.get((class_name, entry.name))
                    if meta:
                        patient_id = meta["patient_id"]
                        sha256_hash = meta["sha256_rgb224"]
                    else:
                        stem = os.path.splitext(entry.name)[0]
                        patient_id = extract_patient_id(class_name, stem)
                        sha256_hash = compute_rgb224_hash(Path(entry.path))
                    items.append(
                        {
                            "class": class_name,
                            "filename": entry.name,
                            "patient_id": patient_id,
                            "sha256_rgb224": sha256_hash,
                            "current_path": Path(entry.path),
                        }
                    )
    return items, source_mode


def split_class_by_patient(
    class_items: List[dict],
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> Dict[str, List[dict]]:
    """Divide los elementos de una clase agrupados por `patient_id`."""
    if not np.isclose(train_ratio + val_ratio + test_ratio, 1.0):
        raise ValueError("Las proporciones train + val + test deben sumar 1.0")

    n = len(class_items)
    groups = np.array([item["patient_id"] for item in class_items])
    unique_groups = np.unique(groups)
    indices = np.arange(n)

    if len(unique_groups) < 3:
        raise ValueError(
            f"Se requieren al menos 3 pacientes únicos para dividir en train/val/test "
            f"(encontrados: {len(unique_groups)})."
        )

    # 1) Separar train vs temp (val + test)
    holdout_ratio = val_ratio + test_ratio
    gss1 = GroupShuffleSplit(n_splits=1, test_size=holdout_ratio, random_state=seed)
    train_idx, temp_idx = next(gss1.split(indices, groups=groups))

    # 2) Separar temp en val vs test
    temp_groups = groups[temp_idx]
    val_frac_in_temp = val_ratio / holdout_ratio
    if len(np.unique(temp_groups)) < 2:
        raise ValueError("No hay suficientes grupos en holdout para separar val y test.")

    gss2 = GroupShuffleSplit(
        n_splits=1, test_size=(1.0 - val_frac_in_temp), random_state=seed + 1
    )
    rel_val_idx, rel_test_idx = next(gss2.split(temp_idx, groups=temp_groups))
    val_idx = temp_idx[rel_val_idx]
    test_idx = temp_idx[rel_test_idx]

    return {
        "train": [class_items[i] for i in train_idx],
        "val": [class_items[i] for i in val_idx],
        "test": [class_items[i] for i in test_idx],
    }


def verify_manifest_zero_leakage(records: List[dict]) -> None:
    """Verifica que no exista solapamiento de paciente ni de hash entre splits."""
    patients_by_split: Dict[str, set] = {s: set() for s in SPLITS}
    hashes_by_split: Dict[str, set] = {s: set() for s in SPLITS}

    for r in records:
        sp = r["split"]
        patients_by_split[sp].add((r["class"], r["patient_id"]))
        hashes_by_split[sp].add(r["sha256_rgb224"])

    pairs = [("train", "val"), ("train", "test"), ("val", "test")]
    for a, b in pairs:
        p_overlap = patients_by_split[a] & patients_by_split[b]
        h_overlap = hashes_by_split[a] & hashes_by_split[b]
        if p_overlap:
            raise AssertionError(
                f"FUGA DETECTADA: {len(p_overlap)} pacientes compartidos entre {a} y {b}: "
                f"{list(p_overlap)[:5]}"
            )
        if h_overlap:
            raise AssertionError(
                f"FUGA DETECTADA: {len(h_overlap)} imágenes idénticas compartidas entre {a} y {b}"
            )


def run_split(
    clean_dir_rel: str,
    processed_dir_rel: str,
    clean_manifest_rel: str,
    dataset_manifest_rel: str,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
    copy_mode: bool,
    verify_only: bool,
) -> None:
    clean_dir = resolve_path(clean_dir_rel, BASE_DIR)
    processed_dir = resolve_path(processed_dir_rel, BASE_DIR)
    clean_manifest = resolve_path(clean_manifest_rel, BASE_DIR)
    dataset_manifest = resolve_path(dataset_manifest_rel, BASE_DIR)

    if verify_only:
        if not dataset_manifest.exists():
            raise FileNotFoundError(f"No existe el manifiesto: {dataset_manifest}")
        with open(dataset_manifest, "r", encoding="utf-8") as f:
            records = list(csv.DictReader(f))
        verify_manifest_zero_leakage(records)
        print(f"AUDITORIA EXITOSA: 0 fugas entre splits en {len(records)} registros.")
        return

    lookup = load_clean_manifest(clean_manifest)
    items, source_mode = collect_items(clean_dir, processed_dir, lookup)

    by_class: Dict[str, List[dict]] = defaultdict(list)
    for it in items:
        by_class[it["class"]].append(it)

    manifest_records: List[dict] = []
    move_plan: List[Tuple[Path, Path]] = []

    print("\n" + "=" * 72)
    print("PARTICIONAMIENTO AGRUPADO POR PACIENTE (70% TRAIN / 15% VAL / 15% TEST)")
    print("=" * 72)

    for class_name in sorted(by_class.keys()):
        class_items = sorted(by_class[class_name], key=lambda x: x["filename"])
        splits_dict = split_class_by_patient(
            class_items, train_ratio, val_ratio, test_ratio, seed
        )

        counts_str = []
        for sp in SPLITS:
            sp_items = splits_dict[sp]
            unique_p = len({it["patient_id"] for it in sp_items})
            counts_str.append(f"{sp}={len(sp_items)} ({unique_p} pac.)")
            for it in sp_items:
                dst = processed_dir / sp / class_name / it["filename"]
                move_plan.append((it["current_path"], dst))
                manifest_records.append(
                    {
                        "class": class_name,
                        "split": sp,
                        "patient_id": it["patient_id"],
                        "filename": it["filename"],
                        "sha256_rgb224": it["sha256_rgb224"],
                        "rel_path": safe_relpath(dst, BASE_DIR),
                    }
                )

        print(f"  [{class_name}] Total={len(class_items)} -> " + " | ".join(counts_str))
        if len(class_items) < 300:
            print(
                f"    AVISO CLINICO: '{class_name}' tiene solo {len(class_items)} imágenes; "
                "priorizar ampliación de datos en el Paso 1."
            )

    # Validar antes de mover archivos
    verify_manifest_zero_leakage(manifest_records)

    if source_mode == "clean":
        for sp in SPLITS:
            sp_dir = processed_dir / sp
            if sp_dir.exists():
                shutil.rmtree(sp_dir)

    print(f"\nEjecutando {'copia' if copy_mode else 'movimiento'} hacia {processed_dir}...")
    for _, dst in move_plan:
        dst.parent.mkdir(parents=True, exist_ok=True)

    from concurrent.futures import ThreadPoolExecutor

    def transfer_one(pair: Tuple[Path, Path]) -> None:
        src, dst = pair
        if src.resolve() == dst.resolve():
            return
        if copy_mode:
            shutil.copy2(src, dst)
        else:
            try:
                os.replace(src, dst)
            except OSError:
                shutil.copy2(src, dst)
                src.unlink()

    with ThreadPoolExecutor(max_workers=8) as ex:
        list(ex.map(transfer_one, move_plan))

    # Limpiar carpetas vacías en data/clean si se movió todo
    if not copy_mode and clean_dir.exists():
        for p in sorted(clean_dir.rglob("*"), reverse=True):
            if p.is_dir():
                try:
                    p.rmdir()
                except OSError:
                    pass
        try:
            clean_dir.rmdir()
        except OSError:
            pass

    dataset_manifest.parent.mkdir(parents=True, exist_ok=True)
    with open(dataset_manifest, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "class",
                "split",
                "patient_id",
                "filename",
                "sha256_rgb224",
                "rel_path",
            ],
        )
        writer.writeheader()
        writer.writerows(manifest_records)

    print("=" * 72)
    print(f"COMPLETADO: {len(manifest_records)} imágenes organizadas con 0% de fugas.")
    print(f"Manifiesto final guardado en: {dataset_manifest}")
    print("=" * 72)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Divide el dataset limpio en train/val/test agrupado por paciente."
    )
    parser.add_argument("--clean-dir", default=DEFAULT_CLEAN_DIR)
    parser.add_argument("--processed-dir", default=DEFAULT_PROCESSED_DIR)
    parser.add_argument("--clean-manifest", default=DEFAULT_CLEAN_MANIFEST)
    parser.add_argument("--dataset-manifest", default=DEFAULT_DATASET_MANIFEST)
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--copy",
        action="store_true",
        help="Copiar en lugar de mover archivos desde data/clean.",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Solo auditar cero fugas sobre dataset_manifest.csv.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_split(
        clean_dir_rel=args.clean_dir,
        processed_dir_rel=args.processed_dir,
        clean_manifest_rel=args.clean_manifest,
        dataset_manifest_rel=args.dataset_manifest,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
        copy_mode=args.copy,
        verify_only=args.verify_only,
    )
