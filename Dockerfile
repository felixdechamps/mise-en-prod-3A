FROM ubuntu:22.04

# Définition du répertoire de travail pour éviter de travailler à la racine
WORKDIR /app

# Installation des dépendances système nécessaires
RUN apt-get update -y && \
    apt-get install -y python3-pip curl && \
    rm -rf /var/lib/apt/lists/*

# Installation de uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin/:$PATH"

# Copie des fichiers de définition des dépendances
# Docker mettra en cache la couche suivante si ces deux fichiers sont inchangés
COPY pyproject.toml uv.lock ./

# Installation des dépendances sans installer le projet lui-même
# Cela permet de gagner du temps lors des modifications de code
RUN uv sync --frozen --no-install-project

# Copie du reste du code source
COPY src ./src
COPY app ./app


# Synchronisation finale pour inclure le projet actuel
RUN uv sync --frozen

# Utilisation de l'environnement virtuel créé par uv
ENV PATH="/app/.venv/bin:$PATH"
CMD ["bash", "-c", "./app/run.sh"]
