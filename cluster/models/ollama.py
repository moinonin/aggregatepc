"""Ollama integration for AggregatePC.

Handles:
- Detecting if Ollama is installed and running
- Starting/stopping Ollama serve
- Pulling models via `ollama pull`
- Loading models into memory via `ollama run`
- Checking which models are already available
"""

from __future__ import annotations

import json
import logging
import subprocess
import time
import os
import socket
from typing import Optional

logger = logging.getLogger("aggregatepc.ollama")

OLLAMA_DEFAULT_PORT = 11434


def is_ollama_installed() -> bool:
    """Check if Ollama CLI is available."""
    try:
        result = subprocess.run(
            ["ollama", "version"],
            capture_output=True, text=True, timeout=10,
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def is_ollama_running() -> bool:
    """Check if Ollama server is currently running."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(2.0)
            s.connect(("127.0.0.1", OLLAMA_DEFAULT_PORT))
            return True
    except (OSError, ConnectionRefusedError):
        return False


def start_ollama_serve() -> bool:
    """Start Ollama serve in the background if not already running."""
    if is_ollama_running():
        logger.info("Ollama is already running")
        return True

    if not is_ollama_installed():
        logger.warning("Ollama is not installed")
        return False

    try:
        subprocess.Popen(
            ["ollama", "serve"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        # Wait for it to start
        for _ in range(15):
            time.sleep(1.0)
            if is_ollama_running():
                logger.info("Ollama server started")
                return True

        logger.warning("Ollama server did not start within 15 seconds")
        return False
    except Exception as e:
        logger.error(f"Failed to start Ollama: {e}")
        return False


def list_ollama_models() -> list[dict]:
    """List models available in Ollama."""
    try:
        result = subprocess.run(
            ["ollama", "list"],
            capture_output=True, text=True, timeout=30,
        )
        if result.returncode != 0:
            return []

        models = []
        for line in result.stdout.strip().splitlines():
            parts = line.split()
            if len(parts) >= 2 and parts[0] != "NAME":
                name = parts[0]
                # Parse size if available
                size_gb = 0
                for part in parts:
                    if "GB" in part:
                        try:
                            size_gb = float(part.replace("GB", ""))
                        except ValueError:
                            pass
                models.append({
                    "name": name,
                    "size_gb": size_gb,
                })
        return models
    except Exception as e:
        logger.error(f"Failed to list Ollama models: {e}")
        return []


def pull_ollama_model(model_name: str, timeout: int = 600) -> bool:
    """Pull a model from Ollama registry.

    Args:
        model_name: Model name (e.g., "llama3", "mistral", "phi")
        timeout: Maximum seconds to wait for pull
    """
    if not is_ollama_installed():
        logger.error("Ollama is not installed")
        return False

    logger.info(f"Pulling model: {model_name}")
    try:
        process = subprocess.Popen(
            ["ollama", "pull", model_name],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )

        start = time.time()
        while time.time() - start < timeout:
            if process.poll() is not None:
                return process.returncode == 0
            time.sleep(2.0)

        process.kill()
        logger.error(f"Pull timed out after {timeout}s")
        return False
    except Exception as e:
        logger.error(f"Failed to pull model: {e}")
        return False


def load_ollama_model(model_name: str) -> bool:
    """Load a model into memory (warm it up).

    Sends a simple prompt to ensure the model is loaded into VRAM.
    """
    try:
        result = subprocess.run(
            ["ollama", "run", model_name, "Hello"],
            capture_output=True, text=True, timeout=60,
        )
        return result.returncode == 0
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return False


def get_best_ollama_model() -> Optional[str]:
    """Get the best available Ollama model name (largest)."""
    models = list_ollama_models()
    if not models:
        return None
    # Sort by size descending
    models.sort(key=lambda m: m.get("size_gb", 0), reverse=True)
    return models[0]["name"] if models else None


def ensure_model_available(model_name: Optional[str] = None) -> Optional[str]:
    """Ensure a model is available and Ollama is running.

    If model_name is None, picks the best available model.
    If no models exist, pulls the specified default.

    Returns the model name that's ready, or None on failure.
    """
    # Start Ollama if needed
    if not start_ollama_serve():
        return None

    # Check existing models
    existing = list_ollama_models()
    if existing:
        if model_name is None:
            # Return largest existing model
            existing.sort(key=lambda m: m.get("size_gb", 0), reverse=True)
            return existing[0]["name"]
        # Check if requested model exists
        if any(m["name"] == model_name for m in existing):
            return model_name

    # Need to pull a model
    if model_name is None:
        model_name = "phi3"  # Default: small, capable model

    if pull_ollama_model(model_name):
        return model_name

    return None


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")

    print(f"Ollama installed: {is_ollama_installed()}")
    print(f"Ollama running: {is_ollama_running()}")

    if is_ollama_installed():
        models = list_ollama_models()
        print(f"Available models: {models}")
        best = get_best_ollama_model()
        print(f"Best model: {best}")
