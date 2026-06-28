"""NAT-friendly worker relay.

Workers keep outbound HTTP polls open to the controller. The controller can
return an inference job in a poll response, and the worker posts the result
back after calling its local Ollama daemon.
"""

from __future__ import annotations

import json
import logging
import queue
import socket
import threading
import time
import uuid
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Optional
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

logger = logging.getLogger("aggregatepc.relay")


def get_local_ip() -> str:
    """Return the local address used for outbound traffic."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("8.8.8.8", 80))
            return s.getsockname()[0]
    except OSError:
        return "127.0.0.1"


@dataclass
class RelayWorker:
    node_id: str
    hardware: dict = field(default_factory=dict)
    models: list[str] = field(default_factory=list)
    status: str = "idle"
    compute_score: float = 0.0
    last_seen: float = field(default_factory=time.time)
    jobs: "queue.Queue[dict]" = field(default_factory=queue.Queue)


class RelayState:
    """In-memory relay state shared by HTTP handlers."""

    def __init__(self) -> None:
        self._workers: dict[str, RelayWorker] = {}
        self._results: dict[str, dict] = {}
        self._lock = threading.Lock()
        self._result_ready = threading.Condition(self._lock)

    def register_worker(self, payload: dict) -> None:
        node_id = payload["node_id"]
        with self._lock:
            worker = self._workers.get(node_id)
            if not worker:
                worker = RelayWorker(node_id=node_id)
                self._workers[node_id] = worker
            worker.hardware = payload.get("hardware", worker.hardware)
            worker.models = payload.get("models", worker.models)
            worker.status = payload.get("status", worker.status)
            worker.compute_score = payload.get("compute_score", worker.compute_score)
            worker.last_seen = time.time()

    def update_worker(self, payload: dict) -> None:
        self.register_worker(payload)

    def get_status(self) -> dict:
        with self._lock:
            workers = []
            now = time.time()
            for worker in self._workers.values():
                workers.append({
                    "node_id": worker.node_id,
                    "status": worker.status,
                    "hardware": worker.hardware,
                    "models": worker.models,
                    "compute_score": worker.compute_score,
                    "last_seen": worker.last_seen,
                    "connected": now - worker.last_seen < 45,
                    "pending_jobs": worker.jobs.qsize(),
                })
            return {
                "workers": workers,
                "worker_count": len(workers),
                "available_count": sum(1 for w in workers if w["connected"]),
            }

    def poll_job(self, node_id: str, timeout: float) -> Optional[dict]:
        with self._lock:
            worker = self._workers.get(node_id)
            if worker:
                worker.last_seen = time.time()
        if not worker:
            return None
        try:
            return worker.jobs.get(timeout=timeout)
        except queue.Empty:
            return None

    def submit_job(self, node_id: str, path: str, body: bytes, headers: dict, timeout: float) -> dict:
        job_id = uuid.uuid4().hex
        job = {
            "job_id": job_id,
            "path": path,
            "body": body.decode("utf-8", errors="replace"),
            "headers": headers,
            "created_at": time.time(),
        }
        with self._lock:
            worker = self._workers.get(node_id)
            if not worker:
                return {"ok": False, "status": 503, "error": f"Relay worker not connected: {node_id}"}
            worker.jobs.put(job)
            deadline = time.time() + timeout
            while job_id not in self._results:
                remaining = deadline - time.time()
                if remaining <= 0:
                    return {"ok": False, "status": 504, "error": f"Timed out waiting for relay worker: {node_id}"}
                self._result_ready.wait(timeout=remaining)
            return self._results.pop(job_id)

    def store_result(self, payload: dict) -> None:
        job_id = payload["job_id"]
        with self._lock:
            self._results[job_id] = payload
            self._result_ready.notify_all()


class RelayHandler(BaseHTTPRequestHandler):
    """HTTP API for workers and local proxy clients."""

    server: "RelayServer"

    def do_GET(self) -> None:
        if self.path != "/status":
            self._write_json(404, {"error": "not found"})
            return
        self._write_json(200, self.server.state.get_status())

    def do_POST(self) -> None:
        payload = self._read_json()
        if payload is None:
            self._write_json(400, {"error": "invalid json"})
            return

        if self.path == "/worker/register":
            self.server.state.register_worker(payload)
            self._write_json(200, {"ok": True})
        elif self.path == "/worker/heartbeat":
            self.server.state.update_worker(payload)
            self._write_json(200, {"ok": True})
        elif self.path == "/worker/poll":
            node_id = payload.get("node_id")
            timeout = min(float(payload.get("timeout", 20)), 30.0)
            job = self.server.state.poll_job(node_id, timeout) if node_id else None
            self._write_json(200, {"job": job})
        elif self.path == "/worker/result":
            self.server.state.store_result(payload)
            self._write_json(200, {"ok": True})
        elif self.path == "/proxy":
            result = self.server.state.submit_job(
                node_id=payload.get("node_id", ""),
                path=payload.get("path", "/api/generate"),
                body=payload.get("body", "").encode(),
                headers=payload.get("headers", {}),
                timeout=float(payload.get("timeout", 120)),
            )
            self._write_json(200, result)
        else:
            self._write_json(404, {"error": "not found"})

    def _read_json(self) -> Optional[dict]:
        try:
            length = int(self.headers.get("Content-Length", "0"))
            return json.loads(self.rfile.read(length).decode())
        except (ValueError, json.JSONDecodeError, UnicodeDecodeError):
            return None

    def _write_json(self, status: int, payload: dict) -> None:
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format, *args) -> None:
        logger.debug(format, *args)


class RelayServer(ThreadingHTTPServer):
    """Threading HTTP server carrying relay state."""

    def __init__(self, server_address: tuple[str, int]):
        super().__init__(server_address, RelayHandler)
        self.state = RelayState()


def start_relay_server(host: str = "0.0.0.0", port: int = 8767) -> RelayServer:
    """Start a relay server in a background thread."""
    server = RelayServer((host, port))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    logger.info("Relay server listening on %s:%s", host, port)
    return server


class RelayWorkerClient:
    """Worker-side outbound relay client."""

    def __init__(
        self,
        controller_address: str,
        relay_port: int,
        node,
        heartbeat_interval_seconds: float = 10.0,
    ) -> None:
        self.controller_address = controller_address
        self.relay_port = relay_port
        self.node = node
        self.heartbeat_interval_seconds = heartbeat_interval_seconds
        self._running = False
        self._thread: Optional[threading.Thread] = None

    @property
    def base_url(self) -> str:
        if self.controller_address.startswith(("http://", "https://")):
            return f"{self.controller_address}:{self.relay_port}" if ":" not in self.controller_address.rsplit("/", 1)[-1] else self.controller_address
        return f"http://{self.controller_address}:{self.relay_port}"

    def start(self) -> None:
        self._running = True
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)

    def _run_loop(self) -> None:
        while self._running:
            try:
                self._register()
                self._poll_once()
            except Exception as e:
                logger.debug("Relay worker loop failed: %s", e)
                time.sleep(self.heartbeat_interval_seconds)

    def _register(self) -> None:
        self._post("/worker/register", self._node_payload(), timeout=5)

    def _poll_once(self) -> None:
        payload = {"node_id": self.node.node_id, "timeout": 20}
        response = self._post("/worker/poll", payload, timeout=30)
        job = response.get("job")
        if job:
            self._handle_job(job)

    def _handle_job(self, job: dict) -> None:
        result = self._execute_ollama_job(job)
        result.update({"node_id": self.node.node_id, "job_id": job["job_id"]})
        self._post("/worker/result", result, timeout=10)

    def _execute_ollama_job(self, job: dict) -> dict:
        url = f"http://127.0.0.1:11434{job.get('path', '/api/generate')}"
        try:
            headers = {
                key: value
                for key, value in job.get("headers", {}).items()
                if key.lower() not in ("host", "content-length", "connection")
            }
            req = Request(url, data=job.get("body", "").encode(), headers=headers)
            with urlopen(req, timeout=120) as resp:
                return {
                    "ok": True,
                    "status": resp.status,
                    "headers": dict(resp.getheaders()),
                    "body": resp.read().decode("utf-8", errors="replace"),
                }
        except HTTPError as e:
            return {
                "ok": False,
                "status": e.code,
                "error": e.read().decode("utf-8", errors="replace"),
            }
        except URLError as e:
            return {"ok": False, "status": 502, "error": str(e.reason)}
        except Exception as e:
            return {"ok": False, "status": 502, "error": str(e)}

    def _node_payload(self) -> dict:
        data = self.node.to_dict()
        return {
            "node_id": self.node.node_id,
            "status": data["status"],
            "hardware": data["hardware"],
            "models": data.get("models", []),
            "compute_score": data.get("compute_score", 0),
        }

    def _post(self, path: str, payload: dict, timeout: float) -> dict:
        body = json.dumps(payload).encode()
        req = Request(
            f"{self.base_url}{path}",
            data=body,
            headers={"Content-Type": "application/json"},
        )
        with urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode())
