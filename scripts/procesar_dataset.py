"""Pipeline idempotente de auditoría, deduplicación y estandarización de imágenes.

Lee desde carpetas origen de solo lectura (por defecto `data/raw`, `Training Data`
y `Testing Data`), selecciona la copia de mayor fidelidad por cada imagen única
(eliminando clones `kaggle_kaggle_...` y duplicados exactos por contenido),
verifica integridad, estandariza a RGB 224x224 JPEG y exporta en `data/clean/`
junto con un manifiesto CSV auditable.

Regla de seguridad: NUNCA escribe sobre las carpetas origen.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import os
import re
import shutil
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
from PIL import Image

if "__file__" in globals():
    BASE_DIR = Path(__file__).resolve().parent.parent
else:
    from google.colab import drive
    drive.mount("/content/drive")
    BASE_DIR = Path("/content/drive/MyDrive/Drive.alumno.udg.mx/Proyectos/lung-pathology-ai")
DEFAULT_SOURCES = ["data/raw", "Training Data", "Testing Data"]
DEFAULT_DEST = "data/clean"
DEFAULT_MANIFEST = "data/clean_manifest.csv"
TARGET_SIZE = (224, 224)
VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
POPCOUNT_LUT = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)


@dataclass(frozen=True)
class CandidateFile:
    class_name: str
    source_root: str
    path: Path
    filename: str
    raw_name: str
    raw_stem: str
    ext: str
    kaggle_depth: int
    test_orig_depth: int


def resolve_path(path_value: str | Path, base: Path = BASE_DIR) -> Path:
    p = Path(path_value)
    return p if p.is_absolute() else (base / p)


def safe_relpath(path: Path, base: Path = BASE_DIR) -> str:
    try:
        return str(path.resolve().relative_to(base.resolve()))
    except ValueError:
        return str(path.resolve())


def strip_legacy_prefixes(filename: str) -> Tuple[str, int, int]:
    """Remueve prefijos acumulados `kaggle_` y `test_orig_` conservando el conteo."""
    name = filename
    kaggle_depth = 0
    test_orig_depth = 0
    while True:
        if name.startswith("kaggle_"):
            name = name[7:]
            kaggle_depth += 1
        elif name.startswith("test_orig_"):
            name = name[10:]
            test_orig_depth += 1
        else:
            break
    return name, kaggle_depth, test_orig_depth


def sanitize_stem(text: str) -> str:
    """Convierte un nombre base en un identificador limpio y seguro para disco."""
    cleaned = re.sub(r"\s+", "_", text.strip())
    cleaned = re.sub(r"[^A-Za-z0-9._-]", "_", cleaned)
    cleaned = re.sub(r"_+", "_", cleaned).strip("_")
    return cleaned or "image"


def extract_patient_id(class_name: str, clean_stem: str) -> str:
    """Infiere un identificador de paciente/grupo para evitar fugas entre splits."""
    stem = re.sub(r"^(cov19db_|covidquex_|rsna_)", "", clean_stem)

    # Kermany / Mooney: person1136_bacteria_3078 -> kermany_person1136
    m_person = re.match(r"^(person\d+)_", stem, flags=re.IGNORECASE)
    if m_person:
        return f"kermany_{m_person.group(1).lower()}"

    # Kermany Normal: NORMAL2-IM-0386-0001 o IM-0122-0001
    m_norm2 = re.match(r"^(NORMAL2-IM-\d+)-", stem, flags=re.IGNORECASE)
    if m_norm2:
        return f"kermany_{m_norm2.group(1).upper()}"

    m_im = re.match(r"^(IM-\d+)-", stem, flags=re.IGNORECASE)
    if m_im:
        return f"kermany_{m_im.group(1).upper()}"

    # COVID-19 Radiography DB: Normal-4760, COVID-123, Viral Pneumonia-45, etc.
    m_rahman = re.match(
        r"^(Normal|COVID|Lung_Opacity|Viral[ _]Pneumonia)-\d+",
        stem,
        flags=re.IGNORECASE,
    )
    if m_rahman:
        return f"rahman_{stem}"

    # Fallback seguro: cada imagen única pertenece a su propio grupo
    cls_tag = sanitize_stem(class_name.lower())
    return f"{cls_tag}_{clean_stem}"


def discover_candidates(source_dirs: Iterable[Path]) -> Dict[str, List[CandidateFile]]:
    """Explora carpetas origen organizadas por clase y construye candidatos."""
    by_class: Dict[str, List[CandidateFile]] = defaultdict(list)

    for src_root in source_dirs:
        if not src_root.exists() or not src_root.is_dir():
            continue
        for class_dir in sorted(src_root.iterdir()):
            if not class_dir.is_dir() or class_dir.name.startswith((".", "_")):
                continue
            class_name = class_dir.name

            # Recorro recursivamente todas las subcarpetas dentro de la clase
            for root, dirs, files in os.walk(class_dir):
                # Excluyo subcarpetas ocultas o auxiliares para no procesar archivos ajenos
                dirs[:] = [
                    d for d in sorted(dirs)
                    if not d.startswith((".", "_")) and d.lower() != "masks"
                ]
                for filename in sorted(files):
                    ext = os.path.splitext(filename)[1].lower()
                    if ext not in VALID_EXTENSIONS:
                        continue
                    raw_name, k_depth, t_depth = strip_legacy_prefixes(filename)
                    raw_stem = os.path.splitext(raw_name)[0]
                    by_class[class_name].append(
                        CandidateFile(
                            class_name=class_name,
                            source_root=src_root.name,
                            path=Path(root) / filename,
                            filename=filename,
                            raw_name=raw_name,
                            raw_stem=raw_stem,
                            ext=ext,
                            kaggle_depth=k_depth,
                            test_orig_depth=t_depth,
                        )
                    )
    return by_class


def select_canonical_candidates(
    candidates: List[CandidateFile],
) -> Tuple[List[Tuple[CandidateFile, str]], int]:
    """Deduplica copias generadas por `kaggle_` conservando la copia de mayor calidad.

    Si existen archivos originales (`kaggle_depth == 0`) con distinta extensión
    bajo el mismo stem (ej. `covid (29).jpeg` y `covid (29).jpg`), conserva todos
    porque pueden ser radiografías distintas; los duplicados reales idénticos se
    filtrarán en el siguiente paso por hash de píxeles.
    """
    grouped: Dict[str, List[CandidateFile]] = defaultdict(list)
    for item in candidates:
        grouped[item.raw_stem].append(item)

    selected: List[Tuple[CandidateFile, str]] = []
    skipped_lineage = 0

    for raw_stem in sorted(grouped.keys()):
        group = grouped[raw_stem]
        originals = [c for c in group if c.kaggle_depth == 0]

        if originals:
            # Conservar originales únicos por extensión (ordenando test_orig_ después del limpio)
            by_ext: Dict[str, List[CandidateFile]] = defaultdict(list)
            for orig in originals:
                by_ext[orig.ext].append(orig)
            need_ext_suffix = len(by_ext) > 1
            for ext, ext_items in sorted(by_ext.items()):
                ext_items.sort(key=lambda c: (c.test_orig_depth, c.filename))
                chosen = ext_items[0]
                out_stem = sanitize_stem(chosen.raw_stem)
                if need_ext_suffix:
                    out_stem = f"{out_stem}__{ext.lstrip('.')}"
                selected.append((chosen, out_stem))
                skipped_lineage += len(ext_items) - 1
            # Cualquier versión kaggle_* del mismo stem es derivada de las originales
            skipped_lineage += len(group) - len(originals)
        else:
            # Solo existen copias `kaggle_+`: tomar la menos recomprimida
            group.sort(key=lambda c: (c.kaggle_depth, c.test_orig_depth, c.filename))
            chosen = group[0]
            out_stem = sanitize_stem(chosen.raw_stem)
            selected.append((chosen, out_stem))
            skipped_lineage += len(group) - 1

    return selected, skipped_lineage


def canonical_priority(filename: str) -> Tuple[int, str]:
    """Prioriza nombres originales con ID clínico real sobre alias heredados."""
    stem = os.path.splitext(filename)[0]
    clean = re.sub(r"^(cov19db_|covidquex_|rsna_)", "", stem)
    if re.match(r"^(person\d+_|NORMAL2-IM-\d+-|IM-\d+-)", clean, flags=re.IGNORECASE):
        return (0, filename)
    if re.match(r"^(Normal|COVID|Lung_Opacity|Viral[ _]Pneumonia)-\d+", clean, flags=re.IGNORECASE):
        return (1, filename)
    return (2, filename)


def compute_perceptual_signature(norm_rgb: Image.Image) -> Tuple[np.ndarray, np.ndarray]:
    """Calcula dHash-256 empaquetado (32 bytes) y miniatura 32x32 uint8."""
    gray = norm_rgb.convert("L")
    g17 = np.asarray(gray.resize((17, 16), Image.Resampling.LANCZOS), dtype=np.int16)
    diff = (g17[:, 1:] > g17[:, :-1]).astype(np.uint8).ravel()
    dhash_bytes = np.packbits(diff)  # shape=(32,) uint8 -> 256 bits
    tiny32 = np.asarray(
        gray.resize((32, 32), Image.Resampling.LANCZOS), dtype=np.uint8
    ).ravel()
    return dhash_bytes, tiny32


def process_single_image(
    task: Tuple[CandidateFile, str, Path, bool],
) -> Tuple[str, Optional[dict], Optional[str]]:
    """Valida, calcula hash canónico + firma perceptual y guarda imagen estandarizada."""
    cand, out_stem, dest_class_dir, dry_run = task
    out_filename = f"{out_stem}.jpg"
    out_path = dest_class_dir / out_filename

    try:
        with Image.open(cand.path) as img:
            img.load()
            orig_w, orig_h = img.size
            orig_mode = img.mode
            orig_fmt = (img.format or "").upper()

            if orig_mode != "RGB":
                rgb_img = img.convert("RGB")
            else:
                rgb_img = img

            if rgb_img.size != TARGET_SIZE:
                norm_img = rgb_img.resize(TARGET_SIZE, Image.Resampling.LANCZOS)
                already_normalized = False
            else:
                norm_img = rgb_img
                already_normalized = (orig_mode == "RGB" and orig_fmt == "JPEG")

            pixel_hash = hashlib.sha256(norm_img.tobytes()).hexdigest()
            dhash_bytes, tiny32 = compute_perceptual_signature(norm_img)

            if not dry_run:
                dest_class_dir.mkdir(parents=True, exist_ok=True)
                if already_normalized:
                    shutil.copy2(cand.path, out_path)
                else:
                    norm_img.save(out_path, format="JPEG", quality=95)

        record = {
            "class": cand.class_name,
            "filename": out_filename,
            "patient_id": extract_patient_id(cand.class_name, out_stem),
            "sha256_rgb224": pixel_hash,
            "dhash256_hex": dhash_bytes.tobytes().hex(),
            "orig_width": orig_w,
            "orig_height": orig_h,
            "orig_mode": orig_mode,
            "source_root": cand.source_root,
            "source_path": safe_relpath(cand.path, BASE_DIR),
            "dest_path": safe_relpath(out_path, BASE_DIR),
            "_abs_dest_path": str(out_path),
            "_dhash_bytes": dhash_bytes,
            "_tiny32": tiny32,
        }
        return "ok", record, None
    except Exception as exc:
        return "error", None, f"{cand.path}: {exc}"


def validate_paths(source_dirs: List[Path], dest_dir: Path) -> None:
    """Impide que el destino coincida o esté contenido dentro de una carpeta origen."""
    dest_resolved = dest_dir.resolve()
    for src in source_dirs:
        src_resolved = src.resolve()
        if dest_resolved == src_resolved or src_resolved in dest_resolved.parents:
            raise ValueError(
                f"Seguridad: la carpeta destino ({dest_resolved}) no puede coincidir "
                f"ni estar dentro de la carpeta origen ({src_resolved})."
            )


def run_pipeline(
    sources: List[str],
    dest: str,
    manifest_path: str,
    workers: int,
    clean_dest: bool,
    dry_run: bool,
) -> None:
    source_dirs = [resolve_path(s, BASE_DIR) for s in sources]
    dest_dir = resolve_path(dest, BASE_DIR)
    manifest_file = resolve_path(manifest_path, BASE_DIR)

    validate_paths(source_dirs, dest_dir)

    print("=" * 64)
    print("LIMPIEZA, DEDUPLICACION Y ESTANDARIZACION DE DATASET")
    print("=" * 64)
    print(f"Origenes activos : {[str(p) for p in source_dirs if p.exists()]}")
    print(f"Destino limpio   : {dest_dir}")
    print(f"Modo simulacion  : {dry_run}")

    if clean_dest and dest_dir.exists() and not dry_run:
        print(f"Limpiando carpeta destino previa: {dest_dir}")
        shutil.rmtree(dest_dir)

    by_class = discover_candidates(source_dirs)
    if not by_class:
        print("No se encontraron imagenes validas en los origenes configurados.")
        return

    all_records: List[dict] = []
    errors: List[str] = []
    seen_hashes_per_class: Dict[str, Dict[str, str]] = defaultdict(dict)
    global_hashes: Dict[str, List[Tuple[str, str]]] = defaultdict(list)
    duplicate_pixel_drops = 0
    duplicate_perceptual_drops = 0

    for class_name, candidates in sorted(by_class.items()):
        selected, skipped_lineage = select_canonical_candidates(candidates)
        print(
            f"\n[{class_name}] Archivos detectados: {len(candidates)} | "
            f"Candidatos unicos por linaje: {len(selected)} | "
            f"Clones descartados por prefijo: {skipped_lineage}",
            flush=True,
        )

        dest_class_dir = dest_dir / class_name
        tasks = [(cand, out_stem, dest_class_dir, dry_run) for cand, out_stem in selected]

        processed = 0
        class_records: List[dict] = []

        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(process_single_image, t) for t in tasks]
            for fut in as_completed(futures):
                status, record, err = fut.result()
                processed += 1
                if status == "error" or record is None:
                    if err:
                        errors.append(err)
                else:
                    class_records.append(record)
                if processed % 500 == 0 or processed == len(tasks):
                    print(f"  Progreso {class_name}: {processed}/{len(tasks)}", flush=True)

        # Deduplicación exacta (SHA-256) + perceptual (dHash-256 + MAE 32x32)
        class_records.sort(key=lambda r: canonical_priority(r["filename"]))
        n_recs = len(class_records)
        acc_dhash = np.empty((n_recs, 32), dtype=np.uint8)
        acc_tiny = np.empty((n_recs, 1024), dtype=np.int16)
        acc_count = 0
        class_perceptual_drops = 0

        for rec in class_records:
            h = rec["sha256_rgb224"]
            abs_dest = Path(rec.pop("_abs_dest_path"))
            dh = rec.pop("_dhash_bytes")
            t32 = rec.pop("_tiny32").astype(np.int16)

            if h in seen_hashes_per_class[class_name]:
                duplicate_pixel_drops += 1
                if not dry_run and abs_dest.exists():
                    abs_dest.unlink()
                continue

            is_perceptual_dup = False
            if acc_count > 0:
                dists = POPCOUNT_LUT[acc_dhash[:acc_count] ^ dh].sum(axis=1)
                close_idx = np.flatnonzero(dists <= 12)
                if close_idx.size > 0:
                    maes = np.mean(np.abs(acc_tiny[close_idx] - t32), axis=1)
                    if float(np.min(maes)) <= 4.5:
                        is_perceptual_dup = True

            if is_perceptual_dup:
                duplicate_perceptual_drops += 1
                class_perceptual_drops += 1
                if not dry_run and abs_dest.exists():
                    abs_dest.unlink()
                continue

            acc_dhash[acc_count] = dh
            acc_tiny[acc_count] = t32
            acc_count += 1
            seen_hashes_per_class[class_name][h] = rec["filename"]
            global_hashes[h].append((class_name, rec["filename"]))
            all_records.append(rec)

        kept_count = len(seen_hashes_per_class[class_name])
        print(
            f"  -> Imagenes finales validas en '{class_name}': {kept_count} "
            f"(duplicados perceptuales eliminados: {class_perceptual_drops})",
            flush=True,
        )

    cross_class_collisions = {
        h: items for h, items in global_hashes.items() if len({c for c, _ in items}) > 1
    }

    if not dry_run and all_records:
        manifest_file.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = list(all_records[0].keys())
        with open(manifest_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_records)

    print("\n" + "=" * 64)
    print("RESUMEN DE SANEAMIENTO")
    print("=" * 64)
    print(f"Total imagenes limpias exportadas     : {len(all_records)}")
    print(f"Duplicados exactos por pixeles        : {duplicate_pixel_drops}")
    print(f"Duplicados perceptuales recompresos   : {duplicate_perceptual_drops}")
    print(f"Colisiones entre clases distintas     : {len(cross_class_collisions)}")
    print(f"Archivos danados/omitidos             : {len(errors)}")
    if cross_class_collisions:
        print("\nALERTA - Imagenes identicas encontradas en distintas clases:")
        for h, items in list(cross_class_collisions.items())[:10]:
            print(f"  - {h[:12]}: {items}")
    if not dry_run:
        print(f"Manifiesto guardado en            : {manifest_file}")
    if errors:
        print("\nPrimeros errores detectados:")
        for e in errors[:10]:
            print(f"  - {e}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audita, deduplica y estandariza radiografías sin sobrescribir el origen."
    )
    parser.add_argument(
        "--sources",
        nargs="+",
        default=DEFAULT_SOURCES,
        help="Carpetas origen relativas al proyecto.",
    )
    parser.add_argument(
        "--dest",
        default=DEFAULT_DEST,
        help="Carpeta destino para imágenes limpias (default: data/clean).",
    )
    parser.add_argument(
        "--manifest",
        default=DEFAULT_MANIFEST,
        help="Ruta del CSV manifiesto generado.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Número de hilos para lectura/procesamiento paralelo.",
    )
    parser.add_argument(
        "--no-clean-dest",
        action="store_true",
        help="No borrar la carpeta destino antes de procesar.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Simula el proceso sin escribir archivos en disco.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_pipeline(
        sources=args.sources,
        dest=args.dest,
        manifest_path=args.manifest,
        workers=args.workers,
        clean_dest=not args.no_clean_dest,
        dry_run=args.dry_run,
    )
