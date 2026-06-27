"""Worker health monitoring and auto-reconnect.

The controller uses this to track worker availability and detect failures.
"""

from __future__ import annotations

import json
import logging
import socket
import time
from dataclasses import dataclass, field
from typing import Optional

from cluster.nodes import Node, NodeRole, NodeStatus

logger = logging.getLogger("aggregatepc.heartbeat")

# If a worker hasn't sent a heartbeat in this long, consider it stale
HEARTBEAT_TIMEOUT_SECONDS = 30.0

# How long to wait before removing a stale worker
STALE_REMOVAL_TIMEOUT_SECONDS = 60.0


@dataclass
class WorkerState:
    """Tracks the state of a connected worker."""
    node: Node
    last_heartbeat: float = field(default_factory=time.time)
    join_time: float = field(default_factory=time.time)
    tasks_completed: int = 0
    consecutive_failures: int = 0

    @property
    def is_stale(self) -> bool:
        return (time.time() - self.last_heartbeat) > HEARTBEAT_TIMEOUT_SECONDS

    @property
    def is_dead(self) -> bool:
        return (time.time() - self.last_heartbeat) > STALE_REMOVAL_TIMEOUT_SECONDS

    def refresh(self) -> None:
        self.last_heartbeat = time.time()
        self.consecutive_failures = 0


class HeartbeatMonitor:
    """Tracks worker health on the controller side."""

    def __init__(self, on_join=None, on_leave=None):
        self._workers: dict[str, WorkerState] = {}
        self._on_join = on_join
        self._on_leave = on_leave

    def register(self, node: Node) -> bool:
        """Register a new worker. Returns True if accepted."""
        is_rejoin = node.node_id in self._workers
        if is_rejoin:
            logger.warning(f"Worker {node.node_id} already registered, refreshing")
            self._workers[node.node_id].refresh()
            return True

        self._workers[node.node_id] = WorkerState(node=node)
        logger.info(f"Registered worker {node.node_id} ({node.hardware.hostname})")
        if self._on_join:
            self._on_join(node)
        return True

    def deregister(self, node_id: str) -> None:
        """Remove a worker from monitoring."""
        if node_id in self._workers:
            node = self._workers[node_id].node
            del self._workers[node_id]
            logger.info(f"Deregistered worker {node_id}")
            if self._on_leave:
                self._on_leave(node)

    def record_heartbeat(self, node_id: str, status: str, compute_score: float) -> None:
        """Record an incoming heartbeat from a worker."""
        if node_id in self._workers:
            state = self._workers[node_id]
            state.refresh()
            state.node.status = NodeStatus(status)
            state.node.last_heartbeat = time.time()

    def heartbeat_timeout(self, timeout: float = HEARTBEAT_TIMEOUT_SECONDS) -> None:
        """Check for stale workers and mark them appropriately."""
        now = time.time()
        for state in self._workers.values():
            elapsed = now - state.last_heartbeat
            if elapsed > STALE_REMOVAL_TIMEOUT_SECONDS:
                state.node.status = NodeStatus.OFFLINE
            elif elapsed > HEARTBEAT_TIMEOUT_SECONDS:
                state.node.status = NodeStatus.BUSY  # Unresponsive but not gone yet

    def get_available_workers(self) -> list[WorkerState]:
        """Get workers that are online/idle and not stale."""
        return [
            s for s in self._workers.values()
            if s.node.status in (NodeStatus.ONLINE, NodeStatus.IDLE)
            and not s.is_stale
        ]

    def get_all_workers(self) -> list[WorkerState]:
        """Get all tracked workers."""
        return list(self._workers.values())

    def prune_dead(self) -> list[str]:
        """Remove workers that haven't sent heartbeats in STALE_REMOVAL_TIMEOUT."""
        dead = [nid for nid, s in self._workers.items() if s.is_dead]
        for nid in dead:
            self.deregister(nid)
        return dead

    @property
    def worker_count(self) -> int:
        return len(self._workers)

    @property
    def available_count(self) -> int:
        return len(self.get_available_workers())


class HeartbeatListener:
    """UDP listener that receives worker heartbeats on the controller."""

    def __init__(self, port: int = 8765, on_join=None, on_leave=None):
        self.port = port
        self._monitor = HeartbeatMonitor(on_join=on_join, on_leave=on_leave)
        self._running = False
        self._socket: Optional[socket.socket] = None
        self._listener_thread: Optional[object] = None

    @property
    def monitor(self) -> HeartbeatMonitor:
        return self._monitor

    def _handle_message(self, data: bytes, addr: tuple[str, int]) -> None:
        """Process an incoming UDP message."""
        try:
            msg = json.loads(data.decode())
            msg_type = msg.get("type", "")

            if msg_type == "join":
                node = Node(
                    node_id=msg["node_id"],
                    role=NodeRole.WORKER,
                    hardware=_hardware_from_dict(msg.get("hardware", {})),
                    address=msg.get("address", addr[0]),
                    status=NodeStatus.IDLE,
                )
                # Attach discovered models to the node
                node.models = msg.get("models", [])
                self._monitor.register(node)
                # Send acknowledgment
                if self._socket:
                    ack = json.dumps({"status": "accepted", "node_id": msg["node_id"]}).encode()
                    self._socket.sendto(ack, addr)

            elif msg_type == "heartbeat":
                self._monitor.record_heartbeat(
                    msg["node_id"],
                    msg.get("status", "idle"),
                    msg.get("compute_score", 0),
                )

            elif msg_type == "leave":
                self._monitor.deregister(msg["node_id"])

            elif msg_type == "status_query":
                # Send back cluster status as JSON
                if self._socket and "status_callback" in msg:
                    callback_addr = (msg["status_callback"]["address"], msg["status_callback"]["port"])
                    status = {
                        "workers": [
                            state.node.to_dict()
                            for state in self._monitor.get_all_workers()
                        ],
                        "worker_count": self._monitor.worker_count,
                        "available_count": self._monitor.available_count,
                    }
                    resp = json.dumps(status).encode()
                    self._socket.sendto(resp, callback_addr)

        except (json.JSONDecodeError, KeyError) as e:
            logger.debug(f"Bad message from {addr}: {e}")

    def _listen_loop(self) -> None:
        """Main listen loop."""
        while self._running:
            try:
                data, addr = self._socket.recvfrom(4096)
                self._handle_message(data, addr)
            except socket.timeout:
                continue
            except OSError:
                if self._running:
                    logger.error("Socket error in heartbeat listener")
                break

    def start(self) -> None:
        """Start listening for worker heartbeats."""
        self._socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._socket.settimeout(2.0)
        self._socket.bind(("", self.port))
        self._running = True

        self._listener_thread = __import__("threading").Thread(
            target=self._listen_loop, daemon=True
        )
        self._listener_thread.start()
        logger.info(f"Heartbeat listener started on port {self.port}")

    def stop(self) -> None:
        """Stop the heartbeat listener."""
        self._running = False
        if self._socket:
            self._socket.close()
        logger.info("Heartbeat listener stopped")


def _hardware_from_dict(data: dict):
    """Reconstruct a HardwareProfile from a heartbeat message."""
    from cluster.detect import CPUInfo, MemoryInfo, GPUInfo
    return __import__("cluster.detect", fromlist=["HardwareProfile"]).HardwareProfile(
        cpu=CPUInfo(
            name=data.get("cpu_name", "unknown"),
            cores_physical=data.get("cpu_cores", 0),
            cores_logical=data.get("cpu_cores", 0),
            architecture="unknown",
        ),
        memory=MemoryInfo(
            total_mb=data.get("ram_mb", 0),
            available_mb=0,
        ),
        gpus=[
            GPUInfo(
                name=g.get("name", ""),
                vram_mb=g.get("vram_mb", 0),
                is_integrated=g.get("integrated", False),
            )
            for g in data.get("gpus", [])
        ],
        hostname=data.get("hostname", ""),
    )
