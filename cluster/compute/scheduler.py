"""Capability-based task scheduler that assigns work to the best-suited nodes."""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Optional

from cluster.compute.gpu_allocator import (
    AllocationPlan,
    GPUAllocator,
    ModelRequirements,
    estimate_model_vram,
)
from cluster.compute.task_queue import Priority, Task, TaskQueue
from cluster.nodes import Node, NodeStatus

logger = logging.getLogger("aggregatepc.scheduler")


@dataclass
class TaskAssignment:
    """Result of scheduling a task to a node."""
    task_id: str
    node_id: str
    allocation: Optional[AllocationPlan]
    status: str  # "assigned", "queued", "failed"
    score: float = 0.0


class TaskScheduler:
    """Assigns tasks to nodes based on capability and availability."""

    def __init__(self):
        self._nodes: dict[str, Node] = {}
        self._task_queue = TaskQueue()
        self._gpu_allocator = GPUAllocator()
        self._lock = threading.Lock()
        self._running = False
        self._assignments: dict[str, TaskAssignment] = {}

    def register_node(self, node: Node) -> None:
        """Register a worker node for task assignment."""
        with self._lock:
            self._nodes[node.node_id] = node
            self._gpu_allocator.update_nodes(list(self._nodes.values()))
            logger.info(f"Registered node {node.node_id} (score={node.compute_score})")

    def deregister_node(self, node_id: str) -> None:
        """Remove a worker node."""
        with self._lock:
            if node_id in self._nodes:
                del self._nodes[node_id]
                self._gpu_allocator.update_nodes(list(self._nodes.values()))
                logger.info(f"Deregistered node {node_id}")

    def submit_task(self, task: Task) -> bool:
        """Submit a task for scheduling."""
        return self._task_queue.submit(task)

    def _score_node_for_task(self, node: Node, task: Task) -> float:
        """Score how well a node fits a task (higher = better)."""
        score = 0.0

        # Check basic resource availability
        if task.required_ram_mb > 0:
            available_ram = node.hardware.memory.total_mb
            if available_ram < task.required_ram_mb:
                return -1  # Cannot run on this node
            score += (available_ram - task.required_ram_mb) / available_ram

        # GPU VRAM scoring
        if task.required_vram_mb > 0:
            available_vram = sum(g.vram_mb for g in node.hardware.gpus)
            if available_vram < task.required_vram_mb:
                return -1  # Cannot run on this node
            # Prefer nodes with closer-fit VRAM (don't waste large GPUs on small tasks)
            vram_ratio = task.required_vram_mb / available_vram
            score += vram_ratio * 100  # Prefer tight fit

        # CPU cores scoring
        if task.required_cpu_cores > 0:
            if node.hardware.cpu.cores_logical < task.required_cpu_cores:
                return -1
            score += node.hardware.cpu.cores_logical - task.required_cpu_cores

        # Prefer idle nodes over busy ones
        if node.status == NodeStatus.IDLE:
            score += 50
        elif node.status == NodeStatus.BUSY:
            score -= 50

        return score

    def _find_best_node(self, task: Task) -> Optional[Node]:
        """Find the best available node for a task."""
        best_node = None
        best_score = -1.0

        for node in self._nodes.values():
            if not node.is_available:
                continue
            score = self._score_node_for_task(node, task)
            if score > best_score:
                best_score = score
                best_node = node

        return best_node

    def schedule_next(self) -> Optional[TaskAssignment]:
        """Schedule the highest-priority pending task."""
        task = self._task_queue.get_next()
        if task is None:
            return None

        node = self._find_best_node(task)
        if node is None:
            # Put task back in queue
            self._task_queue.submit(task)
            return TaskAssignment(
                task_id=task.task_id,
                node_id="",
                allocation=None,
                status="queued",
            )

        # Create allocation plan for GPU tasks
        allocation = None
        if task.required_vram_mb > 0:
            model = ModelRequirements(
                model_name=task.payload.get("model", "unknown"),
                vram_mb=task.required_vram_mb,
                ram_mb=task.required_ram_mb,
                cpu_cores=task.required_cpu_cores,
            )
            allocation = self._gpu_allocator.allocate(model)

        assignment = TaskAssignment(
            task_id=task.task_id,
            node_id=node.node_id,
            allocation=allocation,
            status="assigned",
            score=self._score_node_for_task(node, task),
        )

        with self._lock:
            self._assignments[task.task_id] = assignment

        logger.info(f"Scheduled task {task.task_id} → node {node.node_id}")
        return assignment

    def complete_task(self, task_id: str, result: Any = None) -> bool:
        """Mark a task as completed."""
        with self._lock:
            if task_id in self._assignments:
                del self._assignments[task_id]
        return self._task_queue.complete(task_id, result)

    def get_assignments(self) -> dict[str, TaskAssignment]:
        """Get all current task assignments."""
        with self._lock:
            return dict(self._assignments)

    @property
    def pending_tasks(self) -> int:
        return self._task_queue.pending_count

    @property
    def cluster_capacity(self) -> dict:
        return self._gpu_allocator.get_cluster_capacity()
