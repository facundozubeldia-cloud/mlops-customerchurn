FROM python:3.11-slim

WORKDIR /app

# Copiar dependencias primero (cache de Docker)
COPY requirements-docker.txt .
RUN pip install --no-cache-dir -r requirements-docker.txt

# Copiar el proyecto
COPY src/ ./src/
COPY models/ ./models/

# Exponer puerto de la API
EXPOSE 8000

# Comando de inicio
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
