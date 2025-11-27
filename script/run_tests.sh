#!/usr/bin/env bash

set -eo pipefail

case $@ in
    "unit")
        uv run pytest -m unit -n auto ;;
    "integration")
        uv run pytest -m integration ;;
    *)
        echo "Unknown option $1 (expected unit or integration)" ;;
esac
