"""Controller that aggregates workers and manages the cluster.

The controller is the central coordinator that workers join, tracks their
health, and provides a view of available compute resources.
"""

from __future__ import annotations

import json
import logging
import socket
import time
from dataclasses import dataclass, field
from typing import Optional

from cluster.detect import detect_hardware
from cluster.nodes import Node, NodeRole, NodeStatus
from cluster.network.heartbeat import (
    HeartbeatListener,
    HeartbeatMonitor,
    WorkerState,
)

logger = logging.getLogger("aggregatepc.controller")


@dataclass
class ClusterStats:
    """Summary of cluster state."""
    total_workers: int = 0
    available_workers: int = 0
    total_cpu_cores: int = 0
    total_ram_mb: int = 0
    total_vram_mb: int = 0
    total_compute_score: float = 0.0

    def to_dict(self) -> dict:
        return {
            "total_workers": self.total_workers,
            "available_workers": self.available_workers,
            "total_cpu_cores": self.total_cpu_cores,
            "total_ram_mb": self.total_ram_mb,
            "total_vram_mb": self.total_vram_mb,
            "total_compute_score": round(self.total_compute_score, 2),
        }


class ClusterController:
    """Central controller that aggregates workers and tracks cluster health."""

    def __init__(self, port: int = 8765):
        self.port = port
        self._local_node = self._build_controller_node()
        self._prune_interval = 15.0  # seconds
        self._last_prune = time.time()
        self._running = False

        # Callbacks for worker join/leave events (print to controller terminal)
        def on_join(node: Node) -> None:
            gpus_str = ", ".join(g.name for g in node.hardware.gpus) if node.hardware.gpus else "none"
            print(f"[aggregatepc] + Worker joined: {node.node_id} ({node.hardware.hostname}) "
                  f"- CPU: {node.hardware.cpu.cores_logical}c, RAM: {node.hardware.memory.total_mb}MB, "
                  f"GPU: {gpus_str} [score: {node.compute_score:.1f}]")

        def on_leave(node: Node) -> None:
            print(f"[aggregatepc] - Worker left: {node.node_id} ({node.hardware.hostname})")

        self._heartbeat_listener = HeartbeatListener(port=port, on_join=on_join, on_leave=on_leave)

    def _build_controller_node(self) -> Node:
        """Create a Node representing this controller."""
        hardware = detect_hardware()
        return Node(
            node_id=hardware.hostname,
            role=NodeRole.CONTROLLER,
            hardware=hardware,
            status=NodeStatus.ONLINE,
        )

    @property
    def monitor(self) -> HeartbeatMonitor:
        return self._heartbeat_listener.monitor

    @property
    def local_node(self) -> Node:
        return self._local_node

    def start(self) -> None:
        """Start the controller."""
        self._running = True
        self._heartbeat_listener.start()
        logger.info(f"Controller started on port {self.port}")
        logger.info(f"Controller node: {self._local_node.node_id}")

    def stop(self) -> None:
        """Stop the controller."""
        self._running = False
        self._heartbeat_listener.stop()
        logger.info("Controller stopped")

    def get_stats(self) -> ClusterStats:
        """Get current cluster statistics."""
        monitor = self._heartbeat_listener.monitor
        available = monitor.get_available_workers()
        all_workers = monitor.get_all_workers()

        stats = ClusterStats(
            total_workers=len(all_workers),
            available_workers=len(available),
        )

        for state in all_workers:
            hw = state.node.hardware
            stats.total_cpu_cores += hw.cpu.cores_logical
            stats.total_ram_mb += hw.memory.total_mb
            stats.total_vram_mb += hw.total_vram_mb
            stats.total_compute_score += state.node.compute_score

        return stats

    def get_workers(self) -> list[dict]:
        """Get info about all connected workers."""
        return [
            state.node.to_dict()
            for state in self._heartbeat_listener.monitor.get_all_workers()
        ]

    def get_available_workers(self) -> list[dict]:
        """Get info about currently available workers."""
        return [
            state.node.to_dict()
            for state in self._heartbeat_listener.monitor.get_available_workers()
        ]

    def prune_dead_workers(self) -> list[str]:
        """Remove workers that have stopped sending heartbeats."""
        now = time.time()
        if now - self._last_prune < self._prune_interval:
            return []

        self._last_prune = now
        dead = self._heartbeat_listener.monitor.prune_dead()
        for node_id in dead:
            print(f"[aggregatepc] - Worker removed (timeout): {node_id}")
        return dead

    def run_forever(self) -> None:
        """Run the controller until interrupted."""
        self.start()
        logger.info("Controller running. Press Ctrl+C to stop.")
        try:
            while self._running:
                self.prune_dead_workers()
                time.sleep(5.0)
        except KeyboardInterrupt:
            logger.info("Interrupted — shutting down")
        finally:
            self.stop()

    def status_json(self) -> str:
        """Get cluster status as JSON."""
        stats = self.get_stats()
        return json.dumps({
            "controller": self._local_node.to_dict(),
            "cluster": stats.to_dict(),
            "workers": self.get_workers(),
        }, indent=2)


def get_local_ip() -> str:
    """Get the local IP address."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("8.8.8.8", 80))
            return s.getsockname()[0]
    except OSError:
        return "127.0.0.1"


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")
    import argparse

    parser = argparse.ArgumentParser(description="AggregatePC Controller")
    parser.add_argument("--port", type=int, default=8765, help="UDP port (default: 8765)")
    args = parser.parse_args()

    controller = ClusterController(port=args.port)
    print(f"Controller IP: {get_local_ip()}")
    controller.run_forever()
