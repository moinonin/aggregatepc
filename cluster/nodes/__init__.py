"""Node abstraction layer for cluster members."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from cluster.detect import HardwareProfile, detect_hardware


class NodeRole(Enum):
    CONTROLLER = "controller"
    WORKER = "worker"


class NodeStatus(Enum):
    ONLINE = "online"
    OFFLINE = "offline"
    BUSY = "busy"
    IDLE = "idle"


@dataclass
class Node:
    """Represents a machine in the cluster."""

    node_id: str
    role: NodeRole
    hardware: HardwareProfile
    status: NodeStatus = NodeStatus.IDLE
    last_heartbeat: float = field(default_factory=time.time)
    address: Optional[str] = None  # IPv4 address on the local network
    models: list[str] = field(default_factory=list)  # Models available on this node

    @property
    def is_available(self) -> bool:
        return self.status in (NodeStatus.ONLINE, NodeStatus.IDLE)

    @property
    def compute_score(self) -> float:
        """Rough compute capability score for load balancing."""
        score = self.hardware.cpu.cores_logical * 10
        score += self.hardware.memory.total_mb / 1024  # GB
        for gpu in self.hardware.gpus:
            if not gpu.is_integrated:
                score += gpu.vram_mb / 1024 * 50  # Discrete GPU weighted heavily
            else:
                score += gpu.vram_mb / 1024 * 10
        return score

    def to_dict(self) -> dict:
        return {
            "node_id": self.node_id,
            "role": self.role.value,
            "status": self.status.value,
            "address": self.address,
            "hardware": {
                "hostname": self.hardware.hostname,
                "os": f"{self.hardware.os_name} {self.hardware.os_version}",
                "cpu_name": self.hardware.cpu.name,
                "cpu_cores": self.hardware.cpu.cores_logical,
                "ram_mb": self.hardware.memory.total_mb,
                "gpus": [
                    {"name": g.name, "vram_mb": g.vram_mb, "integrated": g.is_integrated}
                    for g in self.hardware.gpus
                ],
            },
            "compute_score": round(self.compute_score, 2),
            "last_heartbeat": self.last_heartbeat,
        }


def create_local_node(role: NodeRole = NodeRole.WORKER) -> Node:
    """Create a Node representing the local machine with auto-detected hardware."""
    hardware = detect_hardware()
    return Node(
        node_id=hardware.hostname,
        role=role,
        hardware=hardware,
    )


def create_remote_node(
    node_id: str,
    address: str,
    role: NodeRole = NodeRole.WORKER,
    hardware: Optional[HardwareProfile] = None,
) -> Node:
    """Create a Node representing a remote machine (hardware TBD via heartbeat)."""
    if hardware is None:
        # Placeholder; real hardware info comes from heartbeat response
        from cluster.detect import CPUInfo, MemoryInfo
        hardware = HardwareProfile(
            cpu=CPUInfo(name="unknown", cores_physical=0, cores_logical=0, architecture="unknown"),
            memory=MemoryInfo(total_mb=0, available_mb=0),
            hostname=node_id,
        )
    return Node(
        node_id=node_id,
        role=role,
        hardware=hardware,
        address=address,
    )
