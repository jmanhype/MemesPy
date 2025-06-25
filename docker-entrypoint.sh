#!/bin/bash
set -e

# If command starts with 'pytest', just run it
if [[ "$1" == pytest* ]]; then
    exec "$@"
else
    # Otherwise, run the default uvicorn command
    exec python -m uvicorn src.dspy_meme_gen.api.main:app --host 0.0.0.0 --port 8081
fi