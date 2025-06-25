# Build stage
FROM python:3.10-slim AS builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --user --no-cache-dir -r requirements.txt

# Production stage
FROM python:3.10-slim

# Add labels for metadata
ARG BUILD_DATE
ARG VCS_REF
ARG VERSION
LABEL org.opencontainers.image.created=$BUILD_DATE \
      org.opencontainers.image.revision=$VCS_REF \
      org.opencontainers.image.version=$VERSION \
      org.opencontainers.image.title="MemesPy" \
      org.opencontainers.image.description="AI-powered meme generation API" \
      org.opencontainers.image.source="https://github.com/jmanhype/MemesPy"

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=off \
    PIP_DISABLE_PIP_VERSION_CHECK=on

# Create non-root user
RUN useradd -m -u 1000 memespy

WORKDIR /app

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    libpq5 \
    curl \
    netcat-traditional \
    && rm -rf /var/lib/apt/lists/*

# Copy installed packages from builder
COPY --from=builder /root/.local /home/memespy/.local

# Copy application code
COPY --chown=memespy:memespy . .

# Create necessary directories
RUN mkdir -p /var/log/dspy_meme_gen \
    && mkdir -p /var/lib/dspy_meme_gen/meme_storage \
    && mkdir -p static/images/memes \
    && chown -R memespy:memespy /var/log/dspy_meme_gen \
    && chown -R memespy:memespy /var/lib/dspy_meme_gen \
    && chown -R memespy:memespy static

# Switch to non-root user
USER memespy

# Add user's local bin to PATH and set PYTHONPATH
ENV PATH=/home/memespy/.local/bin:$PATH \
    PYTHONPATH=/app/src

# Expose port
EXPOSE 8081

# Copy entrypoint script
COPY --chown=memespy:memespy docker-entrypoint.sh /usr/local/bin/
RUN chmod +x /usr/local/bin/docker-entrypoint.sh

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8081/api/health || exit 1

# Set entrypoint
ENTRYPOINT ["/usr/local/bin/docker-entrypoint.sh"]

# Default command (can be overridden)
CMD [] 