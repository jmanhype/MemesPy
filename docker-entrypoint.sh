#!/bin/bash
set -e

# If command starts with 'pytest', just run it without starting the app
if [[ "$1" == pytest* ]]; then
    # Don't start the app, just run pytest
    exec "$@"
else
    # Otherwise, run the default uvicorn command
    exec python -m uvicorn src.dspy_meme_gen.api.main:app --host 0.0.0.0 --port 8081
fi