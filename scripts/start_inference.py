#!/usr/bin/env python3
"""Cluster inference proxy and model discovery.

Modes:
  1. Local mode: Serve a model locally on this machine
  2. Broadcast mode (--broadcast): Start a proxy that routes to the best model anywhere in the cluster

The proxy mode is the core value: it lets any node inference any model
in the cluster, regardless of which node has the model locally.

Usage:
  python3 scripts/start_inference.py              # Local model serving
  python3 scripts/start_inference.py --broadcast   # Cluster inference proxy
"""

import sys
import os
import time
import socket
import json
import subprocess
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.request import Request, urlopen
from urllib.error import URLError
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging
from cluster.models.registry import discover_all_models, get_best_model
from cluster.config import load_config

logger = logging.getLogger("aggregatepc.inference")


def get_local_ip() -> str:
    """Get the local IP address."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("8.8.8.8", 80))
            return s.getsockname()[0]
    except OSError:
        return "127.0.0.1"


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
            return False

    print(f"[aggregatepc] Loading {model_name} into memory...")
    load_ollama_model(model_name)
    return True


def start_vllm_server(model_path: str, port: int = 8000):
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
                return None
        return process
    except FileNotFoundError:
        return None


class ClusterProxy:
    """Proxy that routes inference requests to the best model in the cluster.

    Architecture:
      - Discovers which nodes have which models
      - Forwards requests to the node that has the best model
      - If no single node can handle it, uses split placement
      - Provides a single OpenAI-compatible endpoint for clients
    """

    def __init__(self, port: int = 8000):
        self.port = port
        self.controller_port = 8765
        self._nodes = []  # [{address, port, models, score}]
        self._best_node = None
        self._best_model = None
        self._lock = threading.Lock()

    def discover_cluster(self, controller_port: int, wait_seconds: int = 5):
        """Discover models across the cluster by querying the controller.

        Uses the controller's status endpoint to get worker info,
        then checks each worker for available Ollama models.
        """
        ollama_port = 11434
        self.controller_port = controller_port
        deadline = time.time() + wait_seconds

        while True:
            # Query controller status to get worker addresses
            status = self._query_controller_status(controller_port)
            nodes = []
            best_node = None
            best_model = None

            if status:
                workers = status.get("workers", [])
                for worker_info in workers:
                    address = worker_info.get("address")
                    if not address:
                        continue

                    # Get models advertised by this worker
                    worker_models = worker_info.get("models", [])

                    # Also check if worker's Ollama is reachable via API
                    ollama_models = self._get_worker_ollama_models(address, ollama_port)
                    backend_reachable = ollama_models is not None

                    # Combine: prefer explicitly advertised models, fall back to Ollama API
                    all_models = sorted(set(worker_models + (ollama_models or [])))

                    nodes.append({
                        "node_id": worker_info.get("node_id", worker_info.get("hostname", "unknown")),
                        "address": address,
                        "models": all_models,
                        "advertised_models": worker_models,
                        "backend_models": ollama_models or [],
                        "backend_reachable": backend_reachable,
                        "compute_score": worker_info.get("compute_score", 0),
                        "ollama_port": ollama_port,
                    })

                # Find best model across cluster
                all_model_infos = []
                model_to_node = {}
                for node in nodes:
                    if not node["backend_reachable"]:
                        continue
                    for model_name in node["models"]:
                        clean_name = model_name.replace("ollama://", "")
                        all_model_infos.append(type("ModelInfo", (), {
                            "name": clean_name,
                            "path": model_name,
                            "size_mb": 0,
                            "model_type": "ollama",
                        }))
                        # Track which node has which model
                        if clean_name not in model_to_node:
                            model_to_node[clean_name] = node

                if all_model_infos:
                    selected = get_best_model(all_model_infos)
                    best_model = selected.name
                    best_node = model_to_node.get(best_model, nodes[0] if nodes else None)

            with self._lock:
                self._nodes = nodes
                self._best_node = best_node
                self._best_model = best_model

            if best_node or time.time() >= deadline:
                return best_node

            time.sleep(1.0)

    def _query_controller_status(self, controller_port: int) -> Optional[dict]:
        """Query the controller's status to get worker info."""
        config = load_config()
        controller_ip = config.get("controller_ip", "127.0.0.1")

        try:
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
                s.settimeout(5.0)
                s.bind(("", 0))
                callback_port = s.getsockname()[1]
                msg = json.dumps({
                    "type": "status_query",
                    "status_callback": {"address": get_local_ip(), "port": callback_port}
                }).encode()
                s.sendto(msg, (controller_ip, controller_port))
                data, _ = s.recvfrom(8192)
                return json.loads(data.decode())
        except Exception as e:
            logger.debug(f"Could not query controller: {e}")
            return None

    def _get_worker_ollama_models(self, worker_address: str, ollama_port: int) -> Optional[list[str]]:
        """Check what Ollama models are available on a worker.

        Returns None when the backend is unreachable. An empty list means the
        backend responded but has no models.
        """
        try:
            url = f"http://{worker_address}:{ollama_port}/api/tags"
            req = Request(url)
            with urlopen(req, timeout=5) as resp:
                data = json.loads(resp.read().decode())
                models = data.get("models", [])
                return [m["name"] for m in models]
        except Exception:
            return None

    def get_status(self) -> dict:
        """Get cluster model status."""
        with self._lock:
            return {
                "nodes": len(self._nodes),
                "best_node": self._best_node["node_id"] if self._best_node else None,
                "best_model": self._best_model,
                "backends": [
                    {
                        "node_id": node["node_id"],
                        "address": node["address"],
                        "reachable": node["backend_reachable"],
                        "advertised_models": node["advertised_models"],
                        "backend_models": node["backend_models"],
                    }
                    for node in self._nodes
                ],
                "available_models": list(set(
                    m.replace("ollama://", "")
                    for node in self._nodes
                    if node["backend_reachable"]
                    for m in node["models"]
                )),
                "all_models": list(set(
                    m.replace("ollama://", "")
                    for node in self._nodes
                    for m in node["models"]
                )),
            }


class ProxyHandler(BaseHTTPRequestHandler):
    """HTTP handler that proxies requests to the best cluster node."""

    def do_GET(self):
        if self.path == "/status":
            status = self.server.proxy.get_status()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(status, indent=2).encode())
        elif self.path == "/v1/models":
            # List available models
            status = self.server.proxy.get_status()
            models_data = {
                "data": [
                    {"id": m, "object": "model", "created": 0, "owned_by": "aggregatepc"}
                    for m in status.get("available_models", [])
                ],
                "object": "list",
            }
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(models_data, indent=2).encode())
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length)

        if self.path in ("/v1/chat/completions", "/api/generate"):
            # Route to the best node
            target = self.server.proxy._best_node
            if not target:
                target = self.server.proxy.discover_cluster(self.server.proxy.controller_port, wait_seconds=2)
            if not target:
                self.send_response(503)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                error = {"error": "No model available in cluster"}
                self.wfile.write(json.dumps(error).encode())
                return

            body = self._with_target_model(body, target)

            # Determine target endpoint
            if self.path == "/v1/chat/completions":
                target_url = f"http://{target['address']}:{target.get('ollama_port', 11434)}/v1/chat/completions"
            else:
                target_url = f"http://{target['address']}:{target.get('ollama_port', 11434)}/api/generate"

            try:
                req = Request(target_url, data=body, headers=dict(self.headers))
                with urlopen(req, timeout=120) as resp:
                    self.send_response(resp.status)
                    for header, value in resp.getheaders():
                        if header.lower() not in ("transfer-encoding", "connection"):
                            self.send_header(header, value)
                    self.end_headers()
                    self.wfile.write(resp.read())
            except Exception as e:
                self.send_response(502)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                error = {"error": f"Backend error: {str(e)}"}
                self.wfile.write(json.dumps(error).encode())
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format, *args):
        # Suppress default logging for cleaner output
        pass

    def _with_target_model(self, body: bytes, target: dict) -> bytes:
        """Use the discovered model when clients send a placeholder model."""
        try:
            payload = json.loads(body.decode())
        except (UnicodeDecodeError, json.JSONDecodeError):
            return body

        requested = payload.get("model")
        target_model = self.server.proxy._best_model
        if requested in (None, "", "any") and target_model:
            payload["model"] = target_model
            return json.dumps(payload).encode()

        if requested not in target.get("models", []) and target_model:
            payload["model"] = target_model
            return json.dumps(payload).encode()

        return body


def start_proxy():
    """Start the cluster inference proxy."""
    config = load_config()
    controller_port = config.get("controller_port", 8765)
    proxy_port = config.get("proxy_port", 8000)

    print("[aggregatepc] Starting cluster inference proxy...")
    print(f"[aggregatepc] Controller port: {controller_port}")
    print(f"[aggregatepc] Proxy port: {proxy_port}")

    # Discover cluster
    proxy = ClusterProxy(port=proxy_port)
    best_node = proxy.discover_cluster(controller_port)

    status = proxy.get_status()

    if best_node:
        print(f"[aggregatepc] Cluster discovered: {status['nodes']} node(s)")
        print(f"[aggregatepc] Best model: {status['best_model']}")
        print(f"[aggregatepc] Running on: {best_node['node_id']} ({best_node['address']})")
    else:
        print("[aggregatepc] No workers/models found — proxy will return 503")
        print("[aggregatepc] Make sure workers are running: aggregatepc worker")

    # Start HTTP proxy
    server = HTTPServer(("0.0.0.0", proxy_port), ProxyHandler)
    server.proxy = proxy

    print(f"[aggregatepc] Proxy ready at http://{get_local_ip()}:{proxy_port}")
    print()
    print("[aggregatepc] Test with:")
    print(f'  curl http://localhost:{proxy_port}/v1/chat/completions -H "Content-Type: application/json" -d \'{{"model":"{status["best_model"] or "any"}","messages":[{{"role":"user","content":"Hello"}}]}}\'')
    print()
    print(f"[aggregatepc] Status: http://localhost:{proxy_port}/status")
    print("[aggregatepc] Press Ctrl+C to stop.")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[aggregatepc] Stopping proxy...")
        server.shutdown()


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
        print()
        print("[aggregatepc] Or start the cluster proxy:")
        print("  make inference")
        sys.exit(1)

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


def main():
    if "--broadcast" in sys.argv:
        start_proxy()
    else:
        local_inference()


if __name__ == "__main__":
    main()
