FROM python:3.11-slim

WORKDIR /app

# Installer les dépendances système minimales (curl pour healthcheck)
RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

# Copier et installer les dépendances Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copier le code
COPY app.py .

# Port d'écoute de Streamlit par défaut
EXPOSE 8501

# Healthcheck : Streamlit expose /_stcore/health
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# Lancer Streamlit
CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501", "--server.headless=true"]