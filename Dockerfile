# Use Python 3.10 slim image as base
FROM python:3.10-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=off \
    PIP_DISABLE_PIP_VERSION_CHECK=on \
    PYSETUP_PATH="/opt/pysetup"

# Set working directory
WORKDIR $PYSETUP_PATH

# Install system dependencies
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        netcat-traditional \
    && rm -rf /var/lib/apt/lists/*

# Copy project files
COPY requirements.txt ./
COPY src/ ./src/
COPY alembic.ini ./
COPY alembic/ ./alembic/

# Install dependencies
RUN pip install -r requirements.txt

# Copy the rest of the application
COPY . .

# Install the application
RUN pip install -e .

# Create necessary directories
RUN mkdir -p /var/log/dspy_meme_gen \
    && mkdir -p /var/lib/dspy_meme_gen/meme_storage

# Set up entrypoint script
COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# Expose port
EXPOSE 8000

# Set entrypoint
ENTRYPOINT ["/entrypoint.sh"]

# Default command
CMD ["uvicorn", "dspy_meme_gen.api.main:app", "--host", "0.0.0.0", "--port", "8000"] 