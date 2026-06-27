#!/usr/bin/env python3
"""Start inference with the best available model on the cluster.

Discovers local models, selects the best one by size, and starts
an appropriate inference server (Ollama, llama.cpp, or vLLM).
"""

import sys
import os
import time
import subprocess
import signal

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cluster.models.registry import discover_all_models, get_best_model


def start_ollama_server():
    """Start Ollama server if not already running."""
    import socket
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(2.0)
            s.connect(("127.0.0.1", 11434))
            return True  # Already running
    except (OSError, ConnectionRefusedError):
        pass

    print("[aggregatepc] Starting Ollama server...")
    try:
        subprocess.Popen(
            ["ollama", "serve"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        for _ in range(15):
            time.sleep(1.0)
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    s.settimeout(2.0)
                    s.connect(("127.0.0.1", 11434))
                    return True
            except (OSError, ConnectionRefusedError):
                continue
    except FileNotFoundError:
        pass
    return False


def start_ollama_model(model_name: str) -> bool:
    """Pull and start serving an Ollama model."""
    from cluster.models.ollama import pull_ollama_model, load_ollama_model

    # Check if model is already available
    from cluster.models.ollama import list_ollama_models
    existing = list_ollama_models()
    if not any(m["name"] == model_name for m in existing):
        print(f"[aggregatepc] Pulling {model_name}...")
        if not pull_ollama_model(model_name):
            print(f"[aggregatepc] Failed to pull {model_name}")
            return False

    print(f"[aggregatepc] Loading {model_name} into memory...")
    load_ollama_model(model_name)
    return True


def start_llama_cpp_server(model_path: str, port: int = 8000) -> subprocess.Popen:
    """Start llama.cpp server for a GGUF model."""
    try:
        process = subprocess.Popen(
            ["llama-server", "-m", model_path, "-c", "2048", "--port", str(port)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return process
    except FileNotFoundError:
        print("[aggregatepc] llama-server not found. Install llama.cpp.")
        return None


def main():
    print("[aggregatepc] Discovering best available model...")

    models = discover_all_models()
    if not models:
        print("[aggregatepc] No models found on this machine.")
        print("[aggregatepc] Pull a model first:")
        print("  ollama pull llama3:8b")
        print("  ollama pull mistral:7b")
        print("  ollama pull phi3:mini")
        print()
        print("[aggregatepc] Or set custom model paths:")
        print("  export GGML_MODELS=/path/to/your/gguf/models")
        print("  export HF_HOME=/path/to/your/huggingface/cache")
        sys.exit(1)

    best = get_best_model(models)
    if not best:
        print("[aggregatepc] Could not determine best model")
        sys.exit(1)

    print(f"[aggregatepc] Best model: {best.name}")
    print(f"[aggregatepc]   Type: {best.model_type}")
    print(f"[aggregatepc]   Size: {best.size_mb}MB")
    print(f"[aggregatepc]   Path: {best.path}")

    if best.model_type == "ollama":
        # Start Ollama and the model
        if not start_ollama_server():
            print("[aggregatepc] Ollama not installed or could not start.")
            print("[aggregatepc] Install Ollama: https://ollama.com/download")
            sys.exit(1)

        if not start_ollama_model(best.name):
            print(f"[aggregatepc] Could not start model {best.name}")
            sys.exit(1)

        print(f"[aggregatepc] Ollama serving {best.name} at http://localhost:11434")
        print(f"[aggregatepc] API endpoint: http://localhost:11434/api/generate")
        print()
        print("[aggregatepc] Test with:")
        print(f'  curl http://localhost:11434/api/generate -d \'{{"model":"{best.name}","prompt":"Hello","stream":false}}\'')
        print()
        print("[aggregatepc] Ollama server is running. Press Ctrl+C to stop.")

        # Keep running
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\n[aggregatepc] Stopping inference server...")

    elif best.model_type == "llama.cpp":
        print(f"[aggregatepc] Starting llama.cpp server for {best.name}...")
        process = start_llama_cpp_server(best.path)
        if process:
            print(f"[aggregatepc] llama.cpp server running on http://localhost:8000")
            print("[aggregatepc] Press Ctrl+C to stop.")
            try:
                process.wait()
            except KeyboardInterrupt:
                process.terminate()
        else:
            print("[aggregatepc] Could not start llama.cpp server")
            sys.exit(1)

    else:
        print(f"[aggregatepc] Model type '{best.model_type}' requires manual setup.")
        print(f"[aggregatepc] For HuggingFace models, use vLLM or transformers:")
        print(f"  python -m vllm.entrypoints.openai.api_server --model {best.path} --port 8000")
        sys.exit(0)


if __name__ == "__main__":
    main()
