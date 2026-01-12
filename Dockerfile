# ~/project-ai-assistant/Dockerfile
FROM python:3.11-slim

# ENV DEBIAN_FRONTEND=noninteractive

# # Install system dependencies
# RUN apt-get update && apt-get install -y \
#     git \
#     build-essential \
#     && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ------------------------------------------------------
# Download HuggingFace models at build time (NO HEREDOC)
# ------------------------------------------------------
COPY download_models.py /app/download_models.py
RUN python /app/download_models.py
# ------------------------------------------------------

# Copy application code
COPY . .

CMD exec gunicorn --bind :$PORT --workers 1 --timeout 0 app:app
