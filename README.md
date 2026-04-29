🚀 Despliegue del Modelo (Containerización y API)
El proyecto utiliza Docker para garantizar la portabilidad y reproducibilidad del modelo de predicción de Churn de AndesLink. El enfoque se divide en dos arquitecturas:

1. Interfaz de Usuario (Dashboard)
Se desarrolló una aplicación web interactiva utilizando Streamlit empaquetada en un contenedor Docker.

Función: Visualización de métricas clave (EDA), análisis de cuartiles y resultados del modelo para usuarios finales.

Puerto: 8501.

2. Service Layer (Inferencia vía API)
Para la integración con sistemas externos, el modelo se expone como un microservicio utilizando FastAPI.

API REST: Permite realizar peticiones de tipo POST (Requests) enviando datos de clientes en formato JSON.

Inferencia en tiempo real: El contenedor carga el modelo entrenado (.pkl) y devuelve la probabilidad de fuga (Response) de forma inmediata.

Puerto: 8000.

🛠️ Flujo de Ejecución (Docker)
Para correr el servicio en cualquier entorno con Docker:

Construcción de la Imagen:

Bash
docker build -t andeslink-ml-service .
Ejecución del Contenedor:

Bash
docker run -p 8000:8000 andeslink-ml-service
Nota: Se incluye un archivo .dockerignore para optimizar el peso de la imagen, excluyendo entornos virtuales (venv), notebooks de experimentación y archivos de caché, asegurando un despliegue ligero y eficiente.