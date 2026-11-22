FROM python:3.11-slim

# Crear usuario sin privilegios root (estándar requerido por Hugging Face Spaces)
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

WORKDIR /home/user/app

# Instalar dependencias de Python
COPY --chown=user:user requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copiar el código fuente y artefactos del modelo
COPY --chown=user:user . .

# Hugging Face Spaces expone el puerto 7860 por defecto
EXPOSE 7860

# Ejecución de la aplicación Streamlit en el puerto 7860
CMD ["streamlit", "run", "src/app.py", "--server.port=7860", "--server.address=0.0.0.0", "--server.headless=true"]
