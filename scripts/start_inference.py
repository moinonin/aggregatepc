#!/usr/bin/env python3
"""Start inference with the best available model.

Modes:
  1. Local mode (default): Discover and serve best model on this machine
  2. Broadcast mode (--broadcast): Find best model across cluster and broadcast to workers

Usage:
  python3 scripts/start_inference.py              # Local inference
  python3 scripts/start_inference.py --broadcast   # Cluster-wide broadcast
"""

import sys
import os
import time
import socket
import json
import subprocess

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cluster.models.registry import discover_all_models, get_best_model
from cluster.config import load_config


def start_ollama_server():
    """Start Ollama server if not already running."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(2.0)
            s.connect(("127.0.0.1", 11434))
            return True
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
    from cluster.models.ollama import pull_ollama_model, load_ollama_model, list_ollama_models

    existing = list_ollama_models()
    if not any(m["name"] == model_name for m in existing):
        print(f"[aggregatepc] Pulling {model_name}...")
        if not pull_ollama_model(model_name):
            print(f"[aggregatepc] Failed to pull {model_name}")
            return False

    print(f"[aggregatepc] Loading {model_name} into memory...")
    load_ollama_model(model_name)
    return True


def start_vllm_server(model_path: str, port: int = 8000) -> subprocess.Popen:
    """Start vLLM OpenAI-compatible server for a HuggingFace model."""
    try:
        process = subprocess.Popen(
            ["python", "-m", "vllm.entrypoints.openai.api_server",
             "--model", model_path, "--port", str(port), "--trust-remote-code"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        for _ in range(30):
            time.sleep(1.0)
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    s.settimeout(1.0)
                    s.connect(("127.0.0.1", port))
                    return process
            except (OSError, ConnectionRefusedError):
                continue
            if process.poll() is not None:
                print("[aggregatepc] vLLM server failed to start")
                return None
        return process
    except FileNotFoundError:
        print("[aggregatepc] vLLM not installed. Install with: pip install vllm")
        return None


def local_inference():
    """Run inference locally on this machine."""
    print("[aggregatepc] Discovering best available model...")

    models = discover_all_models()
    if not models:
        print("[aggregatepc] No models found on this machine.")
        print("[aggregatepc] Pull a model first:")
        print("  ollama pull llama3:8b")
        print("  ollama pull mistral:7b")
        print("  ollama pull phi3:mini")
        sys.exit(1)

    # Prefer Ollama models
    from cluster.models.ollama import is_ollama_installed, list_ollama_models
    ollama_models = list_ollama_models() if is_ollama_installed() else []

    if ollama_models:
        best_ollama = ollama_models[0]
        target_model_name = best_ollama["name"]
        print(f"[aggregatepc] Best model: {target_model_name} (ollama)")
        print(f"[aggregatepc]   Size: {best_ollama.get('size_gb', '?')}GB")

        if not start_ollama_server():
            print("[aggregatepc] Ollama not installed or could not start.")
            print("[aggregatepc] Install Ollama: https://ollama.com/download")
            sys.exit(1)

        if not start_ollama_model(target_model_name):
            print(f"[aggregatepc] Could not start model {target_model_name}")
            sys.exit(1)

        print(f"[aggregatepc] Ollama serving {target_model_name} at http://localhost:11434")
        print(f"[aggregatepc] API endpoint: http://localhost:11434/api/generate")
        print()
        print("[aggregatepc] Test with:")
        print(f'  curl http://localhost:11434/api/generate -d \'{{"model":"{target_model_name}","prompt":"Hello","stream":false}}\'')
        print()
        print("[aggregatepc] Ollama server is running. Press Ctrl+C to stop.")

        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\n[aggregatepc] Stopping inference server...")

    else:
        best = get_best_model(models)
        if not best:
            print("[aggregatepc] Could not determine best model")
            sys.exit(1)

        print(f"[aggregatepc] Best model: {best.name}")
        print(f"[aggregatepc]   Type: {best.model_type}")
        print(f"[aggregatepc]   Size: {best.size_mb}MB")
        print(f"[aggregatepc]   Path: {best.path}")

        if best.model_type == "huggingface":
            print(f"[aggregatepc] Starting vLLM server for {best.name}...")
            process = start_vllm_server(best.path)
            if process:
                print(f"[aggregatepc] vLLM serving {best.name} at http://localhost:8000")
                print(f"[aggregatepc] API endpoint: http://localhost:8000/v1/chat/completions")
                print()
                print("[aggregatepc] Test with:")
                print(f'  curl http://localhost:8000/v1/chat/completions -H "Content-Type: application/json" -d \'{{"model":"{best.name}","messages":[{{"role":"user","content":"Hello"}}]}}\'')
                print()
                print("[aggregatepc] vLLM server is running. Press Ctrl+C to stop.")
                try:
                    process.wait()
                except KeyboardInterrupt:
                    process.terminate()
            else:
                print("[aggregatepc] Could not start vLLM server")
                print("[aggregatepc] Install vLLM: pip install vllm")
                sys.exit(1)
        else:
            print(f"[aggregatepc] Model type '{best.model_type}' requires manual setup.")
            sys.exit(0)


def broadcast_inference():
    """Find best model across cluster and broadcast to all workers."""
    from cluster.network.heartbeat import HeartbeatListener

    config = load_config()
    port = config.get("controller_port", 8765)

    print("[aggregatepc] Discovering best model across cluster...")

    listener = HeartbeatListener(port=port)
    listener.start()
    time.sleep(3)

    # Aggregate models from all workers
    all_models = []
    for state in listener.monitor.get_all_workers():
        for model_name in state.node.models:
            is_ollama = model_name.startswith("ollama://")
            clean_name = model_name.replace("ollama://", "")
            all_models.append(type("ModelInfo", (), {
                "name": clean_name,
                "path": model_name,
                "size_mb": 0,
                "model_type": "ollama" if is_ollama else "unknown",
            }))

    if not all_models:
        print("[aggregatepc] No models found on any worker")
        print("[aggregatepc] Pull a model on a worker first: ollama pull llama3:8b")
        listener.stop()
        sys.exit(1)

    best = get_best_model(all_models)
    if not best:
        print("[aggregatepc] Could not determine best model")
        listener.stop()
        sys.exit(1)

    model_name = best.name
    print(f"[aggregatepc] Cluster best model: {model_name}")

    # Broadcast to all workers
    workers = listener.monitor.get_all_workers()
    if not workers:
        print("[aggregatepc] No workers connected")
        listener.stop()
        sys.exit(1)

    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        msg = json.dumps({"type": "model_select", "model": model_name}).encode()
        for state in workers:
            addr = state.node.address
            if addr:
                s.sendto(msg, (addr, port + 100))
                print(f"[aggregatepc] Sent to {addr} ({state.node.node_id})")

    print(f"[aggregatepc] Broadcast {model_name} to {len(workers)} worker(s)")
    print("[aggregatepc] Workers will pull (if needed) and serve the model.")
    listener.stop()


def main():
    if "--broadcast" in sys.argv:
        broadcast_inference()
    else:
        local_inference()


if __name__ == "__main__":
    main()
