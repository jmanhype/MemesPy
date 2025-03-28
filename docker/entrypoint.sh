#!/bin/bash
set -e

# Function to wait for a service to be ready
wait_for_service() {
    local host="$1"
    local port="$2"
    local service="$3"
    local timeout="${4:-30}"

    echo "Waiting for $service to be ready..."
    for i in $(seq 1 $timeout); do
        if nc -z "$host" "$port"; then
            echo "$service is ready!"
            return 0
        fi
        echo "Waiting for $service... $i/$timeout"
        sleep 1
    done
    echo "Timeout waiting for $service"
    return 1
}

# Wait for PostgreSQL
echo "Waiting for PostgreSQL..."
while ! nc -z db 5432; do
  sleep 1
done
echo "PostgreSQL started"

# Wait for Redis
echo "Waiting for Redis..."
while ! nc -z redis 6379; do
  sleep 1
done
echo "Redis started"

# Run database migrations
echo "Running database migrations..."
alembic upgrade head

# Create initial data if needed
if [ "$APP_ENV" = "development" ]; then
    echo "Creating initial development data..."
    poetry run python -m dspy_meme_gen.cli.db seed
fi

# Start the application
echo "Starting DSPy Meme Generator..."
exec "$@" 