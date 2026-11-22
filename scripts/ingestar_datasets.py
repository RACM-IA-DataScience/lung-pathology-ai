from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

BASE_DIR = Path(__file__).resolve().parent.parent if "__file__" in globals() else Path.cwd()
DOWNLOADS_DIR = BASE_DIR / "data" / "downloads"
RAW_DIR = BASE_DIR / "data" / "raw"
VALID_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def get_kaggle_credentials() -> Tuple[str, str]:
    """Lee usuario y clave desde variables de entorno o `~/.kaggle/kaggle.json`."""
    user = os.environ.get("KAGGLE_USERNAME")
    key = os.environ.get("KAGGLE_KEY")
    if user and key:
        return user, key

    kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
    if kaggle_json.exists():
        data = json.loads(kaggle_json.read_text(encoding="utf-8"))
        return data["username"], data["key"]

    raise RuntimeError(
        "No se detectaron credenciales de Kaggle (`~/.kaggle/kaggle.json` o "
        "variables de entorno KAGGLE_USERNAME/KAGGLE_KEY)."
    )


def _download_zip_with_curl(url: str, target_dir: Path, zip_name: str) -> None:
    target_dir.mkdir(parents=True, exist_ok=True)
    zip_path = target_dir / zip_name
    user, key = get_kaggle_credentials()

    # Selecciono curl.exe en Windows para evitar el error SSL y curl estándar en Linux/Colab
    curl_bin = "curl.exe" if os.name == "nt" else "curl"
    cmd = [
        curl_bin,
        "-L",
        "--fail",
        "-u",
        f"{user}:{key}",
        "-o",
        str(zip_path),
        url,
    ]
    subprocess.run(cmd, check=True)

    print(f"Descomprimiendo {zip_path.name}...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(target_dir)


def covid_radiography_download(target_dir: Path = DOWNLOADS_DIR / "covid19-radiography") -> None:
    _download_zip_with_curl(
        "https://www.kaggle.com/api/v1/datasets/download/tawsifurrahman/covid19-radiography-database",
        target_dir,
        "covid19-radiography-database.zip",
    )


def covid_qu_ex_download(target_dir: Path = DOWNLOADS_DIR / "covid-qu-ex") -> None:
    _download_zip_with_curl(
        "https://www.kaggle.com/api/v1/datasets/download/anasmohammedtahir/covidqu",
        target_dir,
        "covidqu.zip",
    )


def rsna_pneumonia_download(target_dir: Path = DOWNLOADS_DIR / "rsna-pneumonia") -> None:
    _download_zip_with_curl(
        "https://www.kaggle.com/api/v1/competitions/data/download-all/rsna-pneumonia-detection-challenge",
        target_dir,
        "rsna-pneumonia-detection-challenge.zip",
    )


def download_with_kaggle(dataset_key: str, target_dir: Path) -> Path:
    """Descarga y descomprime desde Kaggle usando curl.exe y credenciales locales."""
    downloaders = {
        "covid19-radiography": covid_radiography_download,
        "covid-qu-ex": covid_qu_ex_download,
        "rsna-pneumonia": rsna_pneumonia_download,
    }

    print(f"Descargando '{dataset_key}' en '{target_dir}'...")
    downloaders[dataset_key](target_dir)
    return target_dir


def maybe_extract_zip(folder: Path) -> None:
    """Si el usuario colocó un .zip manual en `data/downloads/`, lo extrae ahí."""
    for zf_path in folder.glob("*.zip"):
        marker = folder / f".extracted_{zf_path.stem}"
        if marker.exists():
            continue
        print(f"Extrayendo archivo manual: {zf_path.name}...")
        with zipfile.ZipFile(zf_path, "r") as zf:
            zf.extractall(folder)
        marker.touch()


def copy_images_with_prefix(
    src_files: List[Path],
    dest_class: str,
    prefix: str,
    dest_raw: Path = RAW_DIR,
    max_items: Optional[int] = None,
) -> int:
    dst_dir = dest_raw / dest_class
    dst_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    selected = src_files[:max_items] if max_items else src_files
    for fp in selected:
        dst_file = dst_dir / f"{prefix}{fp.name}"
        if not dst_file.exists():
            shutil.copy2(fp, dst_file)
        count += 1
    return count


def ingest_covid19_radiography(
    src_root: Path,
    dest_raw: Path,
    max_per_class: Optional[int],
    include_lung_opacity: bool,
    max_normal: Optional[int] = None,
    max_lung_opacity: Optional[int] = None,
    skip_viral: bool = False,
) -> Dict[str, int]:
    """Mapea carpetas del dataset COVID-19 Radiography Database a `dest_raw`."""
    # Busco la raíz real porque puede quedar anidada dentro de `COVID-19_Radiography_Dataset`.
    candidates = [src_root, src_root / "COVID-19_Radiography_Dataset"]
    dataset_root = next((c for c in candidates if (c / "COVID").exists()), None)
    if dataset_root is None:
        raise FileNotFoundError(
            f"No se encontró la estructura esperada (`COVID/`, `Normal/`, `Viral Pneumonia/`) en {src_root}"
        )

    mapping = {
        "COVID": "covid-19",
        "Normal": "Healthy",
    }
    if not skip_viral:
        mapping["Viral Pneumonia"] = "Viral Pneumonia"
    if include_lung_opacity:
        mapping["Lung_Opacity"] = "Bacterial Pneumonia"

    stats: Dict[str, int] = {}
    for folder_name, target_cls in mapping.items():
        img_dir = dataset_root / folder_name / "images"
        if not img_dir.exists():
            img_dir = dataset_root / folder_name
        files = sorted(
            p for p in img_dir.iterdir() if p.is_file() and p.suffix.lower() in VALID_EXTS
        )
        if folder_name == "Normal" and max_normal is not None:
            limit = max_normal
        elif folder_name == "Lung_Opacity" and max_lung_opacity is not None:
            limit = max_lung_opacity
        else:
            limit = max_per_class
        copied = copy_images_with_prefix(
            files, target_cls, prefix="cov19db_", dest_raw=dest_raw, max_items=limit
        )
        stats[f"{folder_name} -> {target_cls}"] = copied
    return stats


def ingest_covid_qu_ex(
    src_root: Path,
    dest_raw: Path,
    max_per_class: Optional[int],
    include_non_covid_as_viral: bool,
) -> Dict[str, int]:
    """Mapea imágenes (excluyendo máscaras) de COVID-QU-Ex hacia `dest_raw`."""
    img_files_by_cls: Dict[str, List[Path]] = {"covid-19": [], "Healthy": [], "Viral Pneumonia": []}

    for split_dir in src_root.rglob("*"):
        if not split_dir.is_dir() or split_dir.name.lower() != "images":
            continue
        parent_name = split_dir.parent.name.lower()
        target_cls = None
        if "covid-19" in parent_name or parent_name == "covid":
            target_cls = "covid-19"
        elif "normal" in parent_name:
            target_cls = "Healthy"
        elif include_non_covid_as_viral and "non-covid" in parent_name:
            target_cls = "Viral Pneumonia"

        if target_cls:
            img_files_by_cls[target_cls].extend(
                p for p in split_dir.iterdir() if p.is_file() and p.suffix.lower() in VALID_EXTS
            )

    stats: Dict[str, int] = {}
    for target_cls, files in img_files_by_cls.items():
        if not files:
            continue
        files.sort()
        copied = copy_images_with_prefix(
            files, target_cls, prefix="covidquex_", dest_raw=dest_raw, max_items=max_per_class
        )
        stats[target_cls] = copied
    return stats


def ingest_rsna_pneumonia(
    src_root: Path, dest_raw: Path, max_per_class: Optional[int]
) -> Dict[str, int]:
    """Convierte DICOMs de adultos de RSNA (`Lung Opacity` vs `Normal`) a JPEG."""
    from concurrent.futures import ThreadPoolExecutor

    try:
        import numpy as np
        import pydicom
        from PIL import Image
        from pydicom.pixel_data_handlers.util import apply_voi_lut
    except ImportError as exc:
        raise ImportError(
            "Para ingerir archivos DICOM de RSNA instala `pydicom` (`pip install pydicom`)."
        ) from exc

    info_csv = src_root / "stage_2_detailed_class_info.csv"
    train_img_dir = src_root / "stage_2_train_images"
    if not info_csv.exists() or not train_img_dir.exists():
        raise FileNotFoundError(
            f"No se encontraron `stage_2_detailed_class_info.csv` y `stage_2_train_images/` en {src_root}"
        )

    patient_to_class: Dict[str, str] = {}
    with open(info_csv, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            pid = row["patientId"]
            cls_detail = row["class"].strip()
            if cls_detail == "Lung Opacity":
                patient_to_class[pid] = "Bacterial Pneumonia"
            elif cls_detail == "Normal" and pid not in patient_to_class:
                patient_to_class[pid] = "Healthy"

    tasks: List[tuple[Path, Path, str]] = []
    counters = {"Bacterial Pneumonia": 0, "Healthy": 0}
    for pid, target_cls in sorted(patient_to_class.items()):
        if max_per_class and counters[target_cls] >= max_per_class:
            continue
        dcm_path = train_img_dir / f"{pid}.dcm"
        if not dcm_path.exists():
            continue
        out_dir = dest_raw / target_cls
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"rsna_{pid}.jpg"
        tasks.append((dcm_path, out_path, target_cls))
        counters[target_cls] += 1

    def convert_one(item: tuple[Path, Path, str]) -> None:
        dcm_path, out_path, _ = item
        if out_path.exists():
            return
        ds = pydicom.dcmread(str(dcm_path))
        arr = apply_voi_lut(ds.pixel_array, ds).astype(np.float32)
        if getattr(ds, "PhotometricInterpretation", "") == "MONOCHROME1":
            arr = arr.max() - arr
        arr -= arr.min()
        max_v = arr.max()
        if max_v > 0:
            arr = (arr / max_v) * 255.0
        img = Image.fromarray(arr.astype(np.uint8)).convert("RGB")
        img.save(out_path, format="JPEG", quality=95)

    with ThreadPoolExecutor(max_workers=8) as ex:
        list(ex.map(convert_one, tasks))

    return counters


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Descarga/ingiere datasets verificados hacia data/raw/."
    )
    parser.add_argument(
        "--dataset",
        choices=["covid19-radiography", "covid-qu-ex", "rsna-pneumonia"],
        required=True,
        help="Dataset a ingestar.",
    )
    parser.add_argument(
        "--source-dir",
        default=None,
        help="Carpeta local ya descargada (por defecto `data/downloads/<dataset>`).",
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="Intentar descarga automática desde Kaggle.",
    )
    parser.add_argument(
        "--max-per-class",
        type=int,
        default=None,
        help="Tope máximo por clase para evitar desbalance extremo.",
    )
    parser.add_argument(
        "--include-lung-opacity-as-bacterial",
        action="store_true",
        help="(Solo covid19-radiography) Mapear Lung_Opacity -> Bacterial Pneumonia.",
    )
    parser.add_argument(
        "--include-non-covid-as-viral",
        action="store_true",
        help="(Solo covid-qu-ex) Mapear Non-COVID -> Viral Pneumonia.",
    )
    parser.add_argument(
        "--max-normal",
        type=int,
        default=None,
        help="(Solo covid19-radiography) Tope específico para Normal/Healthy.",
    )
    parser.add_argument(
        "--max-lung-opacity",
        type=int,
        default=None,
        help="(Solo covid19-radiography) Tope específico para Lung_Opacity -> Bacterial Pneumonia.",
    )
    parser.add_argument(
        "--skip-viral",
        action="store_true",
        help="(Solo covid19-radiography) Omitir Viral Pneumonia (100%% redundante con Kermany).",
    )
    parser.add_argument(
        "--dest-raw",
        default=str(RAW_DIR),
        help="Carpeta destino donde guardar imágenes mapeadas por clase (default: data/raw).",
    )
    args = parser.parse_args()

    dest_raw = Path(args.dest_raw)
    src_dir = (
        Path(args.source_dir)
        if args.source_dir
        else DOWNLOADS_DIR / args.dataset
    )

    if args.download:
        download_with_kaggle(args.dataset, src_dir)
    elif src_dir.exists():
        maybe_extract_zip(src_dir)
    else:
        raise FileNotFoundError(
            f"No existe '{src_dir}'. Coloca ahí los archivos/ZIP descargados o usa `--download`."
        )

    if args.dataset == "covid19-radiography":
        stats = ingest_covid19_radiography(
            src_dir,
            dest_raw,
            args.max_per_class,
            args.include_lung_opacity_as_bacterial,
            max_normal=args.max_normal,
            max_lung_opacity=args.max_lung_opacity,
            skip_viral=args.skip_viral,
        )
    elif args.dataset == "covid-qu-ex":
        stats = ingest_covid_qu_ex(
            src_dir, dest_raw, args.max_per_class, args.include_non_covid_as_viral
        )
    else:
        stats = ingest_rsna_pneumonia(src_dir, dest_raw, args.max_per_class)

    print(f"\nIngesta hacia '{dest_raw}' completada:")
    for k, v in stats.items():
        print(f" - {k}: {v} imágenes")
    print("\nSiguientes pasos obligatorios para deduplicar y particionar sin fugas:")
    print("  1) python scripts/procesar_dataset.py")
    print("  2) python scripts/reorganizar_dataset.py")


if __name__ == "__main__":
    main()