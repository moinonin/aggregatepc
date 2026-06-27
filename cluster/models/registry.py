"""Model registry — discover and track which models are available on each node.

Supports common LLM model storage locations:
  - HuggingFace cache (~/.cache/huggingface/)
  - Ollama models (~/.ollama/models/)
  - llama.cpp models (common paths)
  - Custom paths via environment
"""

from __future__ import annotations

import json
import os
import glob
import logging
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger("aggregatepc.models")


@dataclass
class ModelInfo:
    """Information about a model available on a node."""
    name: str
    path: str
    size_mb: int
    model_type: str  # "huggingface", "ollama", "llama.cpp", "unknown"
    parameters: str = ""  # e.g., "7b", "13b", "70b"
    quantization: str = ""  # e.g., "fp16", "int8", "int4", "Q4_K_M"

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "path": self.path,
            "size_mb": self.size_mb,
            "model_type": self.model_type,
            "parameters": self.parameters,
            "quantization": self.quantization,
        }


def _dir_size_mb(path: str) -> int:
    """Calculate total size of a directory in MB."""
    total = 0
    try:
        for dirpath, _, filenames in os.walk(path):
            for f in filenames:
                fp = os.path.join(dirpath, f)
                if os.path.isfile(fp):
                    total += os.path.getsize(fp)
    except (OSError, PermissionError):
        pass
    return max(total // (1024 * 1024), 1)


def _parse_model_name(dirname: str) -> tuple[str, str, str]:
    """Extract base model name, parameters, and quantization from directory name.

    Examples:
        "llama-7b" -> ("llama-7b", "7b", "")
        "Meta-Llama-3-8B-Instruct" -> ("Meta-Llama-3-8B-Instruct", "8b", "")
        "Q4_K_M" in name -> ("name", "params", "Q4_K_M")
    """
    import re
    name = dirname

    # Try to extract parameter count
    param_match = re.search(r"(\d+(?:\.\d+)?)[Bb]", name)
    parameters = param_match.group(1).lower() + "b" if param_match else ""

    # Try to extract quantization
    quant_patterns = ["int4", "int8", "fp16", "Q4_K_M", "Q5_K_M", "Q8_0", "Q2_K", "Q3_K", "Q6_K"]
    quantization = ""
    for pattern in quant_patterns:
        if pattern.lower() in name.lower():
            quantization = pattern
            break

    return name, parameters, quantization


def discover_huggingface_models() -> list[ModelInfo]:
    """Discover models in the HuggingFace cache."""
    models: list[ModelInfo] = []

    # Check common HF cache locations
    hf_cache_paths = [
        os.path.expanduser("~/.cache/huggingface/hub"),
        os.path.expanduser("~/.cache/huggingface/transformers"),
    ]

    # Also check HF_HOME environment variable
    hf_home = os.environ.get("HF_HOME")
    if hf_home:
        hf_cache_paths.insert(0, os.path.join(hf_home, "hub"))

    for cache_path in hf_cache_paths:
        if not os.path.isdir(cache_path):
            continue

        for entry in os.listdir(cache_path):
            entry_path = os.path.join(cache_path, entry)
            if not os.path.isdir(entry_path):
                continue

            # HF cache uses format: models--org--model-name
            if entry.startswith("models--"):
                parts = entry.split("--")
                if len(parts) >= 4:
                    model_name = "/".join(parts[1:3])  # org/model
                else:
                    model_name = entry.replace("models--", "").replace("--", "/")

                size_mb = _dir_size_mb(entry_path)
                name, parameters, quantization = _parse_model_name(model_name)

                models.append(ModelInfo(
                    name=name,
                    path=entry_path,
                    size_mb=size_mb,
                    model_type="huggingface",
                    parameters=parameters,
                    quantization=quantization,
                ))

    return models


def discover_ollama_models() -> list[ModelInfo]:
    """Discover models managed by Ollama using the ollama list command."""
    models: list[ModelInfo] = []

    try:
        result = subprocess.run(
            ["ollama", "list"],
            capture_output=True, text=True, timeout=30,
        )
        if result.returncode == 0:
            for line in result.stdout.strip().splitlines():
                parts = line.split()
                if len(parts) < 2 or parts[0] == "NAME":
                    continue
                name = parts[0]
                # Parse size
                size_mb = 0
                for i, part in enumerate(parts):
                    if part in ("GB", "MB") and i > 0:
                        try:
                            size_val = float(parts[i - 1])
                            if part == "GB":
                                size_mb = int(size_val * 1024)
                            elif part == "MB":
                                size_mb = int(size_val)
                        except (ValueError, IndexError):
                            pass
                        break

                models.append(ModelInfo(
                    name=name,
                    path=f"ollama://{name}",
                    size_mb=max(size_mb, 1),
                    model_type="ollama",
                ))
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    manifest_models = _discover_ollama_manifest_models()
    seen = {model.name.lower(): model for model in models}
    for model in manifest_models:
        if model.name.lower() not in seen:
            models.append(model)

    return models


def _discover_ollama_manifest_models() -> list[ModelInfo]:
    """Discover Ollama models from local manifests when the daemon is offline."""
    models: list[ModelInfo] = []
    models_root = os.environ.get("OLLAMA_MODELS", os.path.expanduser("~/.ollama/models"))
    manifests_root = os.path.join(models_root, "manifests")
    blobs_root = os.path.join(models_root, "blobs")

    if not os.path.isdir(manifests_root):
        return models

    for root, _, files in os.walk(manifests_root):
        for filename in files:
            manifest_path = os.path.join(root, filename)
            model_name = _ollama_name_from_manifest_path(manifests_root, manifest_path)
            if not model_name:
                continue

            size_mb = _ollama_manifest_size_mb(manifest_path, blobs_root)
            name, parameters, quantization = _parse_model_name(model_name)
            models.append(ModelInfo(
                name=name,
                path=f"ollama://{model_name}",
                size_mb=size_mb,
                model_type="ollama",
                parameters=parameters,
                quantization=quantization,
            ))

    return models


def _ollama_name_from_manifest_path(manifests_root: str, manifest_path: str) -> str:
    """Convert an Ollama manifest path into a model tag like llama3:latest."""
    rel_parts = Path(os.path.relpath(manifest_path, manifests_root)).parts
    if len(rel_parts) < 3:
        return ""

    tag = rel_parts[-1]
    model = rel_parts[-2]
    namespace = rel_parts[-3]
    if namespace == "library":
        return f"{model}:{tag}"
    return f"{namespace}/{model}:{tag}"


def _ollama_manifest_size_mb(manifest_path: str, blobs_root: str) -> int:
    """Calculate model size from manifest layers, falling back to file size."""
    try:
        with open(manifest_path, "r") as f:
            manifest = json.load(f)
    except (OSError, json.JSONDecodeError):
        return 1

    total = 0
    for layer in manifest.get("layers", []):
        digest = layer.get("digest", "")
        if not digest:
            continue

        blob_name = digest.replace(":", "-")
        blob_path = os.path.join(blobs_root, blob_name)
        if os.path.isfile(blob_path):
            total += os.path.getsize(blob_path)
        else:
            total += int(layer.get("size", 0) or 0)

    return max(total // (1024 * 1024), 1)


def discover_llama_cpp_models(paths: list[str] | None = None) -> list[ModelInfo]:
    """Discover GGUF models in common llama.cpp paths."""
    models: list[ModelInfo] = []

    search_paths = paths or [
        os.path.expanduser("~/models"),
        os.path.expanduser("~/.local/share/models"),
        "/usr/local/share/models",
    ]

    # Also check environment variable
    ggml_models = os.environ.get("GGML_MODELS") or os.environ.get("LLAMA_CPP_MODELS")
    if ggml_models:
        search_paths.insert(0, ggml_models)

    gguf_extensions = {".gguf", ".ggml"}

    for search_path in search_paths:
        if not os.path.isdir(search_path):
            continue

        for root, _, files in os.walk(search_path):
            for filename in files:
                filepath = os.path.join(root, filename)
                ext = os.path.splitext(filename)[1].lower()
                if ext in gguf_extensions:
                    size_mb = max(os.path.getsize(filepath) // (1024 * 1024), 1)
                    name, parameters, quantization = _parse_model_name(filename)
                    models.append(ModelInfo(
                        name=name,
                        path=filepath,
                        size_mb=size_mb,
                        model_type="llama.cpp",
                        parameters=parameters,
                        quantization=quantization,
                    ))

    return models


def discover_all_models(extra_paths: list[str] | None = None) -> list[ModelInfo]:
    """Discover all models available on this node.

    Searches:
    1. HuggingFace cache
    2. Ollama models
    3. llama.cpp / GGUF models
    4. Extra paths (custom)
    """
    models: list[ModelInfo] = []

    models.extend(discover_huggingface_models())
    models.extend(discover_ollama_models())
    models.extend(discover_llama_cpp_models(extra_paths))

    # Deduplicate by name, preferring larger (more complete) entries
    seen: dict[str, ModelInfo] = {}
    for model in models:
        key = model.name.lower()
        if key not in seen or model.size_mb > seen[key].size_mb:
            seen[key] = model

    return list(seen.values())


def get_best_model(models: list[ModelInfo]) -> Optional[ModelInfo]:
    """Select the best model from available models.

    Selection criteria (in order of priority):
    1. Prefer Ollama models (already served, fastest to use)
    2. Prefer models that fit in available VRAM
    3. Among those, prefer the largest model (most capable)

    Returns None if no models are available.
    """
    if not models:
        return None

    # Check for Ollama models first (they're already running)
    ollama_models = [m for m in models if m.model_type == "ollama"]
    if ollama_models:
        # Return the largest Ollama model
        return max(ollama_models, key=lambda m: m.size_mb)

    # Get available VRAM from hardware
    try:
        from cluster.detect import detect_hardware
        hw = detect_hardware()
        available_vram_mb = sum(g.vram_mb for g in hw.gpus if not g.is_integrated)
        # If no discrete GPU, use system RAM as fallback (assume we can offload)
        if available_vram_mb == 0:
            available_vram_mb = hw.memory.total_mb
    except Exception:
        available_vram_mb = 0

    # Filter models that fit in VRAM (with 1GB overhead for KV cache)
    fitting_models = [m for m in models if m.size_mb + 1024 <= available_vram_mb]

    if fitting_models:
        # Return the largest model that fits
        return max(fitting_models, key=lambda m: m.size_mb)

    # If nothing fits VRAM, return the largest model anyway (CPU fallback)
    return max(models, key=lambda m: m.size_mb)


def get_model_summary(models: list[ModelInfo]) -> dict:
    """Get a summary of available models."""
    total_size_mb = sum(m.size_mb for m in models)
    by_type: dict[str, int] = {}
    for m in models:
        by_type[m.model_type] = by_type.get(m.model_type, 0) + 1

    best = get_best_model(models)

    return {
        "total_models": len(models),
        "total_size_mb": total_size_mb,
        "total_size_gb": round(total_size_mb / 1024, 2),
        "by_type": by_type,
        "best_model": best.to_dict() if best else None,
        "models": [m.to_dict() for m in models],
    }


if __name__ == "__main__":
    import json
    models = discover_all_models()
    summary = get_model_summary(models)
    print(json.dumps(summary, indent=2))
