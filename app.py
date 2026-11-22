"""Punto de entrada compatible para Hugging Face Spaces y Streamlit.

Desarrollado por: Roberto Alejandro Chagra Martínez (RACM AI)
"""

from pathlib import Path
import sys

BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

# Ejecución del módulo principal de la aplicación
app_path = BASE_DIR / "src" / "app.py"
with open(app_path, "r", encoding="utf-8") as f:
    code = compile(f.read(), str(app_path), "exec")
    exec(code, {"__name__": "__main__", "__file__": str(app_path)})
