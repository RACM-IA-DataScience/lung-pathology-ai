"""Micro-servicio HTTP de inferencia y Grad-CAM para clientes externos (Flutter, Web, Desktop).

Desarrollado para: Roberto Alejandro Chagra Martínez
Disciplina: Artificial Intelligence & Data Science (RACM AI)
"""

import base64
import io
import json
import logging
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import numpy as np
from PIL import Image
import tensorflow as tf

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

BASE_DIR = Path(__file__).resolve().parent.parent

# Cargar configuración de clases
CONFIG_PATH = BASE_DIR / "config" / "mapeo_clases.json"
if CONFIG_PATH.exists():
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        CLASES_DICT = json.load(f)
else:
    CLASES_DICT = {"0": "Bacterial Pneumonia", "1": "Healthy", "2": "Viral Pneumonia", "3": "covid-19"}

# Cargar modelo TensorFlow / Keras
MODEL_PATH = BASE_DIR / "models" / "modelo_chest_disease.keras"
model = None
MODEL_META = None

try:
    from src.model_utils import (
        IMG_SIZE,
        compute_gradcam_heatmap,
        get_backbone_specs,
        load_model_metadata,
        overlay_gradcam,
        preprocess_pil_image,
    )
except ImportError:
    from model_utils import (
        IMG_SIZE,
        compute_gradcam_heatmap,
        get_backbone_specs,
        load_model_metadata,
        overlay_gradcam,
        preprocess_pil_image,
    )

def init_model():
    global model, MODEL_META
    if MODEL_PATH.exists():
        logging.info(f"Cargando modelo neuronal desde {MODEL_PATH}...")
        model = tf.keras.models.load_model(MODEL_PATH)
        MODEL_META = load_model_metadata(BASE_DIR / "config", model=model)
        logging.info("Modelo neuronal y metadatos cargados exitosamente.")
    else:
        logging.warning(f"No se encontró el modelo en {MODEL_PATH}")

class InferenceRequestHandler(BaseHTTPRequestHandler):
    def _send_cors_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")

    def do_OPTIONS(self):
        self.send_response(200)
        self._send_cors_headers()
        self.end_headers()

    def do_GET(self):
        if self.path == "/health" or self.path == "/":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self._send_cors_headers()
            self.end_headers()
            response = {
                "status": "ready" if model is not None else "model_not_found",
                "creator": "Roberto Alejandro Chagra Martínez",
                "brand": "RACM AI - Artificial Intelligence & Data Science",
                "backbone": MODEL_META.get("backbone", "densenet121") if MODEL_META else "densenet121",
                "classes": list(CLASES_DICT.values()),
            }
            self.wfile.write(json.dumps(response).encode("utf-8"))
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        if self.path != "/predict":
            self.send_response(404)
            self.end_headers()
            return

        if model is None:
            self.send_response(503)
            self.send_header("Content-Type", "application/json")
            self._send_cors_headers()
            self.end_headers()
            self.wfile.write(json.dumps({"error": "Modelo no cargado en el servidor."}).encode("utf-8"))
            return

        try:
            content_length = int(self.headers.get("Content-Length", 0))
            body_bytes = self.rfile.read(content_length)
            payload = json.loads(body_bytes.decode("utf-8"))

            image_b64 = payload.get("image_b64", "")
            if "," in image_b64:
                image_b64 = image_b64.split(",", 1)[1]

            image_data = base64.b64decode(image_b64)
            pil_image = Image.open(io.BytesIO(image_data))

            method = payload.get("method", "gradcam++")
            alpha = float(payload.get("alpha", 0.42))
            cmap_name = payload.get("cmap", "jet")
            target_class = payload.get("class_index", None)
            if target_class is not None and int(target_class) < 0:
                target_class = None

            backbone = MODEL_META.get("backbone", "densenet121") if MODEL_META else "densenet121"
            img_size = tuple(MODEL_META.get("img_size", list(IMG_SIZE))) if MODEL_META else IMG_SIZE

            tensor_imagen = preprocess_pil_image(pil_image, backbone=backbone, img_size=img_size)
            heatmap, explained_idx, predicciones = compute_gradcam_heatmap(
                model=model,
                preprocessed_tensor=tensor_imagen,
                class_index=target_class,
                method=method,
            )

            overlay_img = overlay_gradcam(pil_image, heatmap, alpha=alpha, cmap_name=cmap_name)

            buffered = io.BytesIO()
            overlay_img.save(buffered, format="JPEG", quality=90)
            overlay_b64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

            indice_pred = int(np.argmax(predicciones))
            clase_detectada = CLASES_DICT[str(indice_pred)]
            clase_explicada = CLASES_DICT[str(explained_idx)]
            confianza = float(predicciones[indice_pred] * 100.0)
            es_sano = clase_detectada.lower() in {"healthy", "normal"}

            probs_dict = {
                CLASES_DICT[str(i)]: float(predicciones[i])
                for i in range(len(CLASES_DICT))
            }

            resp = {
                "success": True,
                "clase_detectada": clase_detectada,
                "clase_explicada": clase_explicada,
                "confianza": confianza,
                "es_sano": es_sano,
                "metodo_gradcam": method,
                "probabilidades": probs_dict,
                "overlay_b64": overlay_b64,
            }

            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self._send_cors_headers()
            self.end_headers()
            self.wfile.write(json.dumps(resp).encode("utf-8"))

        except Exception as exc:
            logging.error(f"Error durante inferencia: {exc}", exc_info=True)
            self.send_response(500)
            self.send_header("Content-Type", "application/json")
            self._send_cors_headers()
            self.end_headers()
            self.wfile.write(json.dumps({"error": str(exc)}).encode("utf-8"))

def run_server(port: int = 8000):
    init_model()
    server_address = ("0.0.0.0", port)
    httpd = HTTPServer(server_address, InferenceRequestHandler)
    print("=" * 60)
    print(f"🚀 Servidor de Inferencia y Grad-CAM RACM iniciado en http://0.0.0.0:{port}")
    print("   Disponible en tu red local:")
    print(f"   - Local:    http://127.0.0.1:{port}")
    print(f"   - Red Wi-Fi: http://192.168.100.15:{port}")
    print("   Endpoints disponibles:")
    print("   - GET  /health   -> Estado del modelo y creador")
    print("   - POST /predict  -> Inferencia + Generación de Mapa Grad-CAM")
    print("=" * 60)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nServidor detenido por el usuario.")
        httpd.server_close()

if __name__ == "__main__":
    import sys
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
    run_server(port)
