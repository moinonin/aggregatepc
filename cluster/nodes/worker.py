"""Lightweight worker daemon for idle PCs.

Runs on each contributing machine, monitors system usage, and only accepts
compute tasks when the machine is idle (user not actively using it).
"""

import json
import logging
import os
import socket
import subprocess
import threading
import time
from dataclasses import dataclass, field
from typing import Optional

from cluster.detect import detect_hardware
from cluster.nodes import Node, NodeRole, NodeStatus

logger = logging.getLogger("aggregatepc.worker")


@dataclass
class IdleThreshold:
    """Configuration for when a machine is considered 'idle' and available."""
    cpu_percent_max: float = 25.0       # CPU usage must be below this
    memory_percent_max: float = 75.0    # Memory usage must be below this
    check_interval_seconds: float = 5.0 # How often to check
    idle_duration_seconds: float = 30.0 # Must be idle this long before accepting work


@dataclass
class WorkerConfig:
    worker: IdleThreshold = field(default_factory=IdleThreshold)
    controller_port: int = 8765
    heartbeat_interval_seconds: float = 10.0
    advertise_on_network: bool = True


def _get_cpu_usage() -> float:
    """Get current CPU usage percentage (0-100)."""
    try:
        if os.name == "posix":
            with open("/proc/stat", "r") as f:
                line = f.readline()
            fields = line.split()
            idle = int(fields[4])
            total = sum(int(x) for x in fields[1:])
            return (1 - idle / total) * 100 if total > 0 else 0.0
        else:
            import psutil
            return psutil.cpu_percent(interval=0.5)
    except Exception:
        return 100.0


def _get_memory_usage() -> float:
    """Get current memory usage percentage (0-100)."""
    try:
        if os.name == "posix" and os.path.exists("/proc/meminfo"):
            meminfo = {}
            with open("/proc/meminfo", "r") as f:
                for line in f:
                    parts = line.split(":")
                    if len(parts) == 2:
                        val = parts[1].strip().split()[0]
                        meminfo[parts[0].strip()] = int(val)
            total = meminfo.get("MemTotal", 0)
            available = meminfo.get("MemAvailable", 0)
            if total > 0:
                return (1 - available / total) * 100
        else:
            import psutil
            return psutil.virtual_memory().percent
    except Exception:
        return 100.0
    return 0.0


def _is_idle(config: IdleThreshold) -> bool:
    """Check if the machine is currently idle based on thresholds."""
    cpu = _get_cpu_usage()
    mem = _get_memory_usage()
    return cpu < config.cpu_percent_max and mem < config.memory_percent_max


class WorkerDaemon:
    """Lightweight daemon that contributes compute when the PC is idle."""

    def __init__(self, config: Optional[WorkerConfig] = None):
        self.config = config or WorkerConfig()
        self.node = self._build_node()
        self._running = False
        self._idle_since: Optional[float] = None
        self._heartbeat_thread: Optional[threading.Thread] = None
        self._idle_monitor_thread: Optional[threading.Thread] = None
        self._controller_address: Optional[str] = None

    @property
    def is_available(self) -> bool:
        return self.node.status in (NodeStatus.ONLINE, NodeStatus.IDLE)

    @staticmethod
    def _build_node() -> Node:
        """Create a Node representing this worker using auto-detected hardware."""
        hardware = detect_hardware()
        return Node(
            node_id=hardware.hostname,
            role=NodeRole.WORKER,
            hardware=hardware,
            status=NodeStatus.IDLE,
        )

    def _heartbeat_loop(self) -> None:
        """Periodically send heartbeat to controller."""
        while self._running:
            if self._controller_address:
                self._send_heartbeat()
            time.sleep(self.config.heartbeat_interval_seconds)

    def _idle_monitor_loop(self) -> None:
        """Monitor system usage and update availability status."""
        while self._running:
            currently_idle = _is_idle(self.config.worker)

            if currently_idle:
                if self._idle_since is None:
                    self._idle_since = time.time()
                    logger.info("Machine entered idle state")

                elapsed = time.time() - self._idle_since
                if elapsed >= self.config.worker.idle_duration_seconds:
                    if self.node.status != NodeStatus.IDLE:
                        self.node.status = NodeStatus.IDLE
                        logger.info("Machine confirmed idle — available for tasks")
            else:
                if self._idle_since is not None:
                    logger.info("Machine no longer idle (user active)")
                self._idle_since = None
                if self.node.status != NodeStatus.BUSY:
                    self.node.status = NodeStatus.BUSY

            time.sleep(self.config.worker.check_interval_seconds)

    def _send_heartbeat(self) -> None:
        """Send heartbeat message to the controller."""
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
                s.settimeout(2.0)
                msg = json.dumps({
                    "type": "heartbeat",
                    "node_id": self.node.node_id,
                    "status": self.node.status.value,
                    "compute_score": self.node.compute_score,
                    "address": self._get_local_ip(),
                    "models": self.node.models,
                }).encode()
                s.sendto(msg, (self._controller_address, self.config.controller_port))
        except Exception as e:
            logger.debug(f"Heartbeat send failed: {e}")

    def _get_local_ip(self) -> str:
        """Get local IP address."""
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
                s.connect(("8.8.8.8", 80))
                return s.getsockname()[0]
        except OSError:
            return "127.0.0.1"

    def join(self, controller_address: str) -> bool:
        """Join a cluster by contacting the controller."""
        self._controller_address = controller_address

        try:
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
                s.settimeout(5.0)
                msg = json.dumps({
                    "type": "join",
                    "node_id": self.node.node_id,
                    "hardware": self.node.to_dict()["hardware"],
                    "address": self._get_local_ip(),
                    "models": [],  # Will be updated after Ollama starts
                }).encode()
                s.sendto(msg, (controller_address, self.config.controller_port))

                # Wait for acknowledgment
                data, _ = s.recvfrom(1024)
                response = json.loads(data.decode())
                return response.get("status") == "accepted"
        except Exception as e:
            logger.error(f"Failed to join cluster: {e}")
            return False

    def _advertise_models(self) -> None:
        """Re-discover and advertise models to the controller after Ollama starts."""
        from cluster.models.registry import discover_all_models

        local_models = discover_all_models()
        model_names = [m.name for m in local_models]

        # Fallback: if discover_all_models returns empty, check Ollama API directly
        if not model_names:
            try:
                from cluster.models.ollama import list_ollama_models, is_ollama_installed
                if is_ollama_installed():
                    ollama_models = list_ollama_models()
                    model_names = [m["name"] for m in ollama_models]
            except Exception:
                pass

        # Last resort: check if there are model blobs in ~/.ollama/models/blobs/
        if not model_names:
            try:
                blobs_dir = os.path.expanduser("~/.ollama/models/blobs")
                if os.path.isdir(blobs_dir) and os.listdir(blobs_dir):
                    # There are model blobs but Ollama daemon might not be running
                    # Still report that models exist (Ollama can load them on demand)
                    model_names = ["ollama-blobs-available"]
                    logger.info(f"Found model blobs in {blobs_dir} but Ollama daemon not responding")
            except Exception:
                pass

        # Update the node's models
        self.node.models = model_names

        if model_names:
            logger.info(f"Updated models: {', '.join(model_names)}")

        # Send a heartbeat with updated models
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
                s.settimeout(2.0)
                msg = json.dumps({
                    "type": "heartbeat",
                    "node_id": self.node.node_id,
                    "status": self.node.status.value,
                    "compute_score": self.node.compute_score,
                    "address": self._get_local_ip(),
                    "models": model_names,
                }).encode()
                s.sendto(msg, (self._controller_address, self.config.controller_port))
        except Exception as e:
            logger.debug(f"Failed to advertise models: {e}")

    def start(self) -> None:
        """Start the worker daemon."""
        self._running = True
        logger.info(f"Starting worker {self.node.node_id}")

        # Start idle monitor thread
        self._idle_monitor_thread = threading.Thread(
            target=self._idle_monitor_loop, daemon=True
        )
        self._idle_monitor_thread.start()

        # Start heartbeat thread
        self._heartbeat_thread = threading.Thread(
            target=self._heartbeat_loop, daemon=True
        )
        self._heartbeat_thread.start()

        # Auto-start Ollama and serve best model
        self._start_ollama_service()

        # After Ollama is running, advertise models to controller
        threading.Thread(target=self._delayed_advertise, daemon=True).start()

        logger.info("Worker daemon started")

    def _delayed_advertise(self) -> None:
        """Wait for Ollama to start, then advertise models."""
        # Wait longer and retry to ensure Ollama daemon is ready
        for attempt in range(3):
            time.sleep(5)
            self._advertise_models()
            if self.node.models:
                break

    def _start_ollama_service(self) -> None:
        """Start Ollama and load the best available model in a background thread."""
        def _ollama_setup():
            try:
                from cluster.models.ollama import (
                    is_ollama_installed,
                    start_ollama_serve,
                    get_best_ollama_model,
                    ensure_model_available,
                )
                from cluster.models.registry import discover_all_models, get_best_model

                if not is_ollama_installed():
                    logger.info("Ollama not installed — skipping model serving")
                    return

                # Start Ollama server
                if not start_ollama_serve():
                    logger.warning("Could not start Ollama server")
                    return

                # Check for best model (Ollama first, then others)
                all_models = discover_all_models()
                best = get_best_model(all_models)
                if best:
                    if best.model_type == "ollama":
                        logger.info(f"Best model (ollama): {best.name}")
                    else:
                        ollama_best = get_best_ollama_model()
                        if ollama_best:
                            logger.info(f"Best model (ollama): {ollama_best}")
                        else:
                            logger.info(f"Best model: {best.name} (type: {best.model_type})")
                else:
                    logger.info("No models found locally")

            except Exception as e:
                logger.debug(f"Ollama setup error: {e}")

        thread = threading.Thread(target=_ollama_setup, daemon=True)
        thread.start()

    def stop(self) -> None:
        """Stop the worker daemon."""
        self._running = False
        self.node.status = NodeStatus.OFFLINE
        if self._heartbeat_thread:
            self._heartbeat_thread.join(timeout=5.0)
        if self._idle_monitor_thread:
            self._idle_monitor_thread.join(timeout=5.0)
        logger.info("Worker daemon stopped")

    def run_forever(self) -> None:
        """Run the worker until interrupted."""
        self.start()
        try:
            while self._running:
                time.sleep(1.0)
        except KeyboardInterrupt:
            logger.info("Interrupted — shutting down")
        finally:
            self.stop()


def create_worker_node() -> Node:
    """Create a worker Node for this machine."""
    hardware = detect_hardware()
    return Node(
        node_id=hardware.hostname,
        role=NodeRole.WORKER,
        hardware=hardware,
        status=NodeStatus.IDLE,
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")
    import argparse

    parser = argparse.ArgumentParser(description="AggregatePC Worker Daemon")
    parser.add_argument("--controller", type=str, default=None, help="Controller IP address")
    parser.add_argument("--controller-port", type=int, default=8765, help="Controller UDP port (default: 8765)")
    parser.add_argument("--cpu-threshold", type=float, default=25.0, help="Max CPU % to be considered idle")
    parser.add_argument("--mem-threshold", type=float, default=75.0, help="Max memory % to be considered idle")
    parser.add_argument("--idle-duration", type=float, default=30.0, help="Seconds of idle before accepting work")
    args = parser.parse_args()

    config = WorkerConfig(
        controller_port=args.controller_port,
        worker=IdleThreshold(
            cpu_percent_max=args.cpu_threshold,
            memory_percent_max=args.mem_threshold,
            idle_duration_seconds=args.idle_duration,
        )
    )

    daemon = WorkerDaemon(config=config)

    if args.controller:
        if daemon.join(args.controller):
            print(f"Joined controller at {args.controller}")
        else:
            print("Failed to join controller")
            exit(1)

    daemon.run_forever()
