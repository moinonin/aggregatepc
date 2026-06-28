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
import errno
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


def get_proxy_host(config: dict) -> str:
    """Return the IP address the inference proxy should bind and advertise."""
    return config.get("proxy_host") or get_local_ip()


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
        env = os.environ.copy()
        env.setdefault("OLLAMA_HOST", "0.0.0.0:11434")
        subprocess.Popen(
            ["ollama", "serve"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            env=env,
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
             "--model", model_path, "--host", "0.0.0.0",
             "--port", str(port), "--trust-remote-code"],
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
        self._model_to_node = {}
        self._advertised_model_to_node = {}
        self._lock = threading.Lock()

    def discover_cluster(self, controller_port: int, wait_seconds: int = 5):
        """Discover models across the cluster by querying the controller.

        Uses the controller's status endpoint to get worker info,
        then checks each worker for available Ollama models.
        """
        config = load_config()
        backend_overrides = config.get("ollama_backends", {})
        relay_workers = _query_relay_workers(config)
        self.controller_port = controller_port
        deadline = time.time() + wait_seconds

        while True:
            # Query controller status to get worker addresses
            status = self._query_controller_status(controller_port)
            nodes = []
            best_node = None
            best_model = None
            model_to_node = {}
            advertised_model_to_node = {}

            if status:
                workers = status.get("workers", [])
            else:
                workers = []

            if not workers and relay_workers:
                workers = [
                    {
                        "node_id": worker["node_id"],
                        "address": None,
                        "advertised_address": None,
                        "models": worker.get("models", []),
                        "hardware": worker.get("hardware", {}),
                        "compute_score": worker.get("compute_score", 0),
                    }
                    for worker in relay_workers.values()
                    if worker.get("connected")
                ]

            if workers:
                for worker_info in workers:
                    backend_candidates = _worker_backend_candidates(worker_info, backend_overrides)
                    node_id = worker_info.get("node_id", worker_info.get("hostname", "unknown"))
                    relay_worker = relay_workers.get(node_id)
                    relay_reachable = bool(relay_worker and relay_worker.get("connected"))
                    if not backend_candidates and not relay_reachable:
                        continue

                    # Get models advertised by this worker
                    worker_models = worker_info.get("models", [])

                    # Also check if worker's Ollama is reachable via API
                    backend, ollama_models = self._find_reachable_ollama(backend_candidates) if backend_candidates else (None, None)
                    backend_reachable = ollama_models is not None
                    selected_backend = backend or (backend_candidates[0] if backend_candidates else {"host": None, "port": 11434})

                    # Combine: prefer explicitly advertised models, fall back to Ollama API
                    relay_models = relay_worker.get("models", []) if relay_worker else []
                    all_models = sorted(set(worker_models + relay_models + (ollama_models or [])))

                    nodes.append({
                        "node_id": node_id,
                        "address": selected_backend["host"],
                        "observed_address": worker_info.get("address"),
                        "advertised_address": worker_info.get("advertised_address"),
                        "candidate_addresses": [_format_backend_candidate(candidate) for candidate in backend_candidates],
                        "models": all_models,
                        "advertised_models": worker_models,
                        "backend_models": ollama_models or [],
                        "backend_reachable": backend_reachable,
                        "relay_reachable": relay_reachable,
                        "relay_url": _relay_url(config),
                        "compute_score": worker_info.get("compute_score", 0),
                        "ollama_port": selected_backend["port"],
                    })
                    for model_name in all_models:
                        clean_name = _normalize_model_name(model_name)
                        if clean_name and clean_name not in advertised_model_to_node:
                            advertised_model_to_node[clean_name] = nodes[-1]

                # Find best model across cluster
                all_model_infos = []
                for node in nodes:
                    if not (node["backend_reachable"] or node["relay_reachable"]):
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
                self._model_to_node = model_to_node
                self._advertised_model_to_node = advertised_model_to_node

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
        except Exception as e:
            logger.debug(f"Could not reach Ollama on {worker_address}:{ollama_port}: {e}")
            return None

    def _find_reachable_ollama(self, backends: list[dict]) -> tuple[Optional[dict], Optional[list[str]]]:
        """Return the first backend where Ollama responds."""
        for backend in backends:
            models = self._get_worker_ollama_models(backend["host"], backend["port"])
            if models is not None:
                return backend, models
        return None, None

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
                        "observed_address": node["observed_address"],
                        "advertised_address": node["advertised_address"],
                        "candidate_addresses": node["candidate_addresses"],
                        "reachable": node["backend_reachable"],
                        "relay_reachable": node["relay_reachable"],
                        "advertised_models": node["advertised_models"],
                        "backend_models": node["backend_models"],
                    }
                    for node in self._nodes
                ],
                "available_models": list(set(
                    m.replace("ollama://", "")
                    for node in self._nodes
                    if node["backend_reachable"] or node["relay_reachable"]
                    for m in node["models"]
                )),
                "all_models": list(set(
                    m.replace("ollama://", "")
                    for node in self._nodes
                    for m in node["models"]
                )),
            }

    def select_target(self, requested_model: Optional[str]) -> tuple[Optional[dict], Optional[str]]:
        """Select a backend for a requested model, falling back to the best model."""
        normalized = _normalize_model_name(requested_model)
        with self._lock:
            if normalized and normalized not in ("any", "auto"):
                return self._model_to_node.get(normalized), normalized

            return self._best_node, self._best_model

    def explain_unavailable_model(self, requested_model: Optional[str]) -> str:
        """Return a concrete reason a requested model cannot currently be routed."""
        normalized = _normalize_model_name(requested_model)
        if not normalized:
            return "No model available in cluster"

        with self._lock:
            node = self._advertised_model_to_node.get(normalized)
            if node and not node["backend_reachable"]:
                if node.get("relay_reachable"):
                    return f"Model is advertised but direct backend is unreachable; relay should handle it: {normalized}"
                return (
                    f"Model is advertised but backend is unreachable: {normalized} "
                    f"on {node['node_id']} ({node['address']}:{node.get('ollama_port', 11434)})"
                )

        return f"Model not available in cluster: {requested_model}"


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
            requested_model = self._requested_model(body)
            target, target_model = self.server.proxy.select_target(requested_model)
            if not target:
                self.server.proxy.discover_cluster(self.server.proxy.controller_port, wait_seconds=2)
                target, target_model = self.server.proxy.select_target(requested_model)
            if not target:
                self.send_response(503)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                error = {"error": self.server.proxy.explain_unavailable_model(requested_model)}
                self.wfile.write(json.dumps(error).encode())
                return

            body = self._with_target_model(body, target, target_model)

            try:
                if target.get("backend_reachable"):
                    # Determine target endpoint
                    if self.path == "/v1/chat/completions":
                        target_url = f"http://{target['address']}:{target.get('ollama_port', 11434)}/v1/chat/completions"
                    else:
                        target_url = f"http://{target['address']}:{target.get('ollama_port', 11434)}/api/generate"

                    req = Request(target_url, data=body, headers=dict(self.headers))
                    with urlopen(req, timeout=120) as resp:
                        self.send_response(resp.status)
                        for header, value in resp.getheaders():
                            if header.lower() not in ("transfer-encoding", "connection"):
                                self.send_header(header, value)
                        self.end_headers()
                        self.wfile.write(resp.read())
                elif target.get("relay_reachable"):
                    result = _submit_relay_job(target, self.path, body, dict(self.headers))
                    status = int(result.get("status", 502))
                    self.send_response(status)
                    self.send_header("Content-Type", "application/json")
                    self.end_headers()
                    if result.get("ok"):
                        self.wfile.write(result.get("body", "").encode())
                    else:
                        self.wfile.write(json.dumps({"error": result.get("error", "Relay backend error")}).encode())
                else:
                    raise RuntimeError("Selected backend is not reachable directly or via relay")
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

    def _requested_model(self, body: bytes) -> Optional[str]:
        """Extract requested model from an OpenAI/Ollama JSON request body."""
        try:
            payload = json.loads(body.decode())
        except (UnicodeDecodeError, json.JSONDecodeError):
            return None
        return payload.get("model")

    def _with_target_model(self, body: bytes, target: dict, target_model: Optional[str]) -> bytes:
        """Use the discovered model when clients send a placeholder model."""
        try:
            payload = json.loads(body.decode())
        except (UnicodeDecodeError, json.JSONDecodeError):
            return body

        requested = payload.get("model")
        if requested in (None, "", "any") and target_model:
            payload["model"] = target_model
            return json.dumps(payload).encode()

        target_models = {_normalize_model_name(model) for model in target.get("models", [])}
        if _normalize_model_name(requested) not in target_models and target_model:
            payload["model"] = target_model
            return json.dumps(payload).encode()

        return body


def _normalize_model_name(model_name: Optional[str]) -> Optional[str]:
    if not model_name:
        return None
    return model_name.replace("ollama://", "")


def _worker_backend_candidates(worker_info: dict, backend_overrides: dict | None = None) -> list[dict]:
    """Return candidate backends to try for a worker, in priority order."""
    backend_overrides = backend_overrides or {}
    node_id = worker_info.get("node_id") or worker_info.get("hostname")
    candidates = []
    override = backend_overrides.get(node_id)
    if override:
        candidates.append(override)

    candidates.extend([
        {"host": worker_info.get("backend_address"), "port": int(worker_info.get("backend_port", 11434))},
        _backend_from_host_port(worker_info.get("ollama_host"), default_port=11434),
        {"host": worker_info.get("address"), "port": 11434},
        {"host": worker_info.get("advertised_address"), "port": 11434},
    ])

    seen = set()
    ordered = []
    for candidate in candidates:
        if not candidate or not candidate.get("host"):
            continue
        key = (candidate["host"], candidate["port"])
        if key in seen:
            continue
        seen.add(key)
        ordered.append({"host": candidate["host"], "port": candidate["port"]})
    return ordered


def _backend_from_host_port(value: Optional[str], default_port: int) -> dict:
    if not value:
        return {"host": None, "port": default_port}
    host = value
    port = default_port
    if ":" in value:
        host_part, _, port_part = value.rpartition(":")
        host = host_part or host
        try:
            port = int(port_part)
        except ValueError:
            port = default_port
    return {"host": host, "port": port}


def _format_backend_candidate(candidate: dict) -> str:
    return f"{candidate['host']}:{candidate['port']}"


def _relay_url(config: dict) -> str:
    controller_ip = config.get("controller_ip", "127.0.0.1")
    relay_port = config.get("relay_port", 8767)
    return f"http://{controller_ip}:{relay_port}"


def _query_relay_workers(config: dict) -> dict[str, dict]:
    """Return relay-connected workers keyed by node_id."""
    try:
        with urlopen(f"{_relay_url(config)}/status", timeout=2) as resp:
            status = json.loads(resp.read().decode())
    except Exception:
        return {}
    return {
        worker.get("node_id"): worker
        for worker in status.get("workers", [])
        if worker.get("node_id")
    }


def _submit_relay_job(target: dict, path: str, body: bytes, headers: dict) -> dict:
    """Submit an inference job through the controller relay."""
    relay_url = target.get("relay_url")
    if not relay_url:
        return {"ok": False, "status": 503, "error": "Relay URL missing"}
    payload = {
        "node_id": target["node_id"],
        "path": path,
        "body": body.decode("utf-8", errors="replace"),
        "headers": {
            key: value
            for key, value in headers.items()
            if key.lower() not in ("host", "content-length", "connection")
        },
        "timeout": 120,
    }
    req = Request(
        f"{relay_url}/proxy",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urlopen(req, timeout=130) as resp:
        return json.loads(resp.read().decode())


def start_proxy():
    """Start the cluster inference proxy."""
    config = load_config()
    controller_port = config.get("controller_port", 8765)
    proxy_port = config.get("proxy_port", 8000)
    proxy_host = get_proxy_host(config)

    print("[aggregatepc] Starting cluster inference proxy...")
    print(f"[aggregatepc] Controller port: {controller_port}")
    print(f"[aggregatepc] Proxy address: {proxy_host}:{proxy_port}")

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
    try:
        server = HTTPServer((proxy_host, proxy_port), ProxyHandler)
    except OSError as e:
        if e.errno in (errno.EADDRINUSE, 48):
            print(f"[aggregatepc] Proxy address already in use: {proxy_host}:{proxy_port}")
            print(f"[aggregatepc] Existing proxy status: http://{proxy_host}:{proxy_port}/status")
            print("[aggregatepc] Stop the existing proxy or set a different proxy port in configs/cluster.conf.")
            sys.exit(1)
        raise
    server.proxy = proxy

    proxy_url = f"http://{proxy_host}:{proxy_port}"
    print(f"[aggregatepc] Proxy ready at {proxy_url}")
    print()
    print("[aggregatepc] Test with:")
    print(f'  curl {proxy_url}/v1/chat/completions -H "Content-Type: application/json" -d \'{{"model":"{status["best_model"] or "any"}","messages":[{{"role":"user","content":"Hello"}}]}}\'')
    print()
    print(f"[aggregatepc] Status: {proxy_url}/status")
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

        local_ip = get_local_ip()
        print(f"[aggregatepc] Ollama serving {target_model_name} at http://{local_ip}:11434")
        print(f"[aggregatepc] API endpoint: http://{local_ip}:11434/api/generate")
        print()
        print("[aggregatepc] Test with:")
        print(f'  curl http://{local_ip}:11434/api/generate -d \'{{"model":"{target_model_name}","prompt":"Hello","stream":false}}\'')
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
                local_ip = get_local_ip()
                print(f"[aggregatepc] vLLM serving {best.name} at http://{local_ip}:8000")
                print(f"[aggregatepc] API endpoint: http://{local_ip}:8000/v1/chat/completions")
                print()
                print("[aggregatepc] Test with:")
                print(f'  curl http://{local_ip}:8000/v1/chat/completions -H "Content-Type: application/json" -d \'{{"model":"{best.name}","messages":[{{"role":"user","content":"Hello"}}]}}\'')
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
