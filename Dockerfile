# Hugging Face Spaces Dockerfile for Rancho Cordova Chatbot
# Uses Docker SDK with CPU Basic (free tier)

FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Create a non-root user (required by HF Spaces)
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

WORKDIR $HOME/app

# Copy requirements and install dependencies
COPY --chown=user:user requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# Download sentence-transformers model at build time
COPY --chown=user:user download_models.py .
RUN python download_models.py

# Copy application code
COPY --chown=user:user . .

# Create directories for persistent data
RUN mkdir -p $HOME/app/src/ranchocordova/chroma_db \
    && mkdir -p $HOME/app/src/ranchocordova/web_cache

# Expose port 7860 (HF Spaces default)
EXPOSE 7860

# Start the Flask app with gunicorn
CMD ["gunicorn", "--bind", "0.0.0.0:7860", "--workers", "1", "--timeout", "120", "app:app"]
