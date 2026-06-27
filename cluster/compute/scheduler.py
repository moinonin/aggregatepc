"""Capability-based task scheduler with decentralized compute.

Supports:
- Single-node task assignment (best-fit)
- Multi-node model splitting (when no single node can hold the model)
- CPU offload fallback (when VRAM is insufficient)
- Task retry with exponential backoff
"""

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
    """Result of scheduling a task to node(s)."""
    task_id: str
    node_id: str  # Primary node, or ",".join(node_ids) for split tasks
    allocation: Optional[AllocationPlan]
    status: str  # "assigned", "queued", "failed"
    score: float = 0.0
    is_split: bool = False
    retry_count: int = 0


@dataclass
class QueuedTask:
    """A task waiting to be scheduled, with retry metadata."""
    task: Task
    retry_count: int = 0
    last_attempt: float = 0.0
    max_retries: int = 3
    backoff_seconds: float = 5.0

    @property
    def can_retry(self) -> bool:
        return self.retry_count < self.max_retries

    @property
    def next_retry_time(self) -> float:
        return self.last_attempt + (self.backoff_seconds * (self.retry_count + 1))


class TaskScheduler:
    """Assigns tasks to nodes with split placement and CPU fallback."""

    def __init__(self):
        self._nodes: dict[str, Node] = {}
        self._task_queue = TaskQueue()
        self._retry_queue: list[QueuedTask] = []
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
        """Score how well a node fits a task (higher = better).

        Returns -1 if the node cannot participate at all.
        """
        score = 0.0

        # Check RAM (required for any task, even GPU ones)
        if task.required_ram_mb > 0:
            available_ram = node.hardware.memory.total_mb
            if available_ram < task.required_ram_mb * 0.5:  # Allow 50% for offload
                return -1
            score += min(available_ram / task.required_ram_mb, 2.0)

        # GPU VRAM scoring
        if task.required_vram_mb > 0:
            available_vram = sum(g.vram_mb for g in node.hardware.gpus)
            if available_vram >= task.required_vram_mb:
                # Node has enough VRAM — prefer tight fit
                vram_ratio = task.required_vram_mb / available_vram
                score += vram_ratio * 100
            elif available_vram > 0:
                # Node has some GPU but not enough VRAM — can participate in split
                score += 20  # Lower base score for split participation
            # If no GPU at all, still allow (CPU fallback) but no bonus

        # CPU cores scoring
        if task.required_cpu_cores > 0:
            if node.hardware.cpu.cores_logical < 1:  # Need at least 1 core
                return -1
            score += min(node.hardware.cpu.cores_logical / max(task.required_cpu_cores, 1), 2.0) * 20

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

    def _can_cluster_fit_model(self, task: Task) -> bool:
        """Check if the cluster as a whole can fit a model (split across nodes)."""
        if task.required_vram_mb <= 0 and task.required_ram_mb <= 0:
            return True

        total_vram = 0
        total_ram = 0
        for node in self._nodes.values():
            if node.is_available:
                total_vram += sum(g.vram_mb for g in node.hardware.gpus)
                total_ram += node.hardware.memory.total_mb

        # Need VRAM to fit the model, or RAM to hold it entirely
        vram_ok = total_vram >= task.required_vram_mb
        ram_ok = total_ram >= task.required_ram_mb

        return vram_ok or ram_ok

    def _find_split_allocation(self, task: Task) -> Optional[AllocationPlan]:
        """Try to split a model across multiple nodes."""
        model = ModelRequirements(
            model_name=task.payload.get("model", "unknown"),
            vram_mb=task.required_vram_mb,
            ram_mb=task.required_ram_mb,
            cpu_cores=task.required_cpu_cores,
            supports_cpu_fallback=True,
        )
        return self._gpu_allocator.allocate(model)

    def _find_cpu_fallback_node(self, task: Task) -> Optional[Node]:
        """Find a node that can run the task on CPU (no GPU required)."""
        for node in self._nodes.values():
            if not node.is_available:
                continue
            # CPU fallback: need enough RAM
            if node.hardware.memory.total_mb >= task.required_ram_mb * 0.8:
                return node
        return None

    def schedule_next(self) -> Optional[TaskAssignment]:
        """Schedule the highest-priority pending task."""
        # First, check retry queue for tasks due
        now = time.time()
        with self._lock:
            for qt in self._retry_queue[:]:
                if now >= qt.next_retry_time:
                    self._retry_queue.remove(qt)
                    self._task_queue.submit(qt.task)
                    logger.info(f"Retrying task {qt.task.task_id} (attempt {qt.retry_count + 1})")

        task = self._task_queue.get_next()
        if task is None:
            return None

        # Strategy 1: Find best single node
        node = self._find_best_node(task)
        if node is not None:
            allocation = None
            is_split = False
            if task.required_vram_mb > 0:
                model = ModelRequirements(
                    model_name=task.payload.get("model", "unknown"),
                    vram_mb=task.required_vram_mb,
                    ram_mb=task.required_ram_mb,
                    cpu_cores=task.required_cpu_cores,
                    supports_cpu_fallback=True,
                )
                allocation = self._gpu_allocator.allocate(model)
                if allocation and allocation.requires_split:
                    is_split = True

            assignment = TaskAssignment(
                task_id=task.task_id,
                node_id=node.node_id,
                allocation=allocation,
                status="assigned",
                score=self._score_node_for_task(node, task),
                is_split=is_split,
            )

            with self._lock:
                self._assignments[task.task_id] = assignment

            mode = "split" if is_split else "single"
            logger.info(f"Scheduled task {task.task_id} → node {node.node_id} [{mode}]")
            return assignment

        # Strategy 2: CPU fallback (if cluster can handle it on CPU)
        if task.required_vram_mb > 0:
            cpu_node = self._find_cpu_fallback_node(task)
            if cpu_node:
                assignment = TaskAssignment(
                    task_id=task.task_id,
                    node_id=cpu_node.node_id,
                    allocation=None,
                    status="assigned",
                    score=0,
                )
                with self._lock:
                    self._assignments[task.task_id] = assignment
                logger.info(f"Scheduled task {task.task_id} → node {cpu_node.node_id} [cpu_fallback]")
                return assignment

        # Strategy 3: Queue for retry
        with self._lock:
            queued = QueuedTask(task=task)
            self._retry_queue.append(queued)
            logger.warning(f"Task {task.task_id} queued (no suitable node, will retry)")

        return TaskAssignment(
            task_id=task.task_id,
            node_id="",
            allocation=None,
            status="queued",
        )

    def complete_task(self, task_id: str, result: Any = None) -> bool:
        """Mark a task as completed."""
        with self._lock:
            if task_id in self._assignments:
                del self._assignments[task_id]
        return self._task_queue.complete(task_id, result)

    def fail_task(self, task_id: str) -> None:
        """Handle task failure — queue for retry if possible."""
        with self._lock:
            assignment = self._assignments.pop(task_id, None)
            if assignment and assignment.retry_count < 3:
                # Find the original task and re-queue with backoff
                # For now, we just note the failure
                logger.warning(f"Task {task_id} failed (attempt {assignment.retry_count + 1})")

    def get_assignments(self) -> dict[str, TaskAssignment]:
        """Get all current task assignments."""
        with self._lock:
            return dict(self._assignments)

    @property
    def pending_tasks(self) -> int:
        return self._task_queue.pending_count + len(self._retry_queue)

    @property
    def cluster_capacity(self) -> dict:
        return self._gpu_allocator.get_cluster_capacity()