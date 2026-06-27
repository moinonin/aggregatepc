"""Task queue with resource requirements and priority ordering."""

from __future__ import annotations

import heapq
import threading
import time
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Optional


class Priority(IntEnum):
    LOW = 3
    NORMAL = 2
    HIGH = 1
    CRITICAL = 0


@dataclass
class Task:
    """A unit of work with resource requirements."""
    task_id: str
    task_type: str  # "llm_inference", "batch_compute", etc.
    priority: Priority = Priority.NORMAL
    required_ram_mb: int = 0
    required_vram_mb: int = 0
    required_cpu_cores: int = 0
    payload: dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    assigned_node: Optional[str] = None
    result: Optional[Any] = None

    def __lt__(self, other: Task) -> bool:
        """Compare for priority queue (lower priority value = higher priority)."""
        if self.priority != other.priority:
            return self.priority < other.priority
        return self.created_at < other.created_at


class TaskQueue:
    """Thread-safe priority queue for compute tasks."""

    def __init__(self):
        self._heap: list[Task] = []
        self._lock = threading.Lock()
        self._task_ids: set[str] = set()
        self._completed: dict[str, Task] = {}

    def submit(self, task: Task) -> bool:
        """Submit a task to the queue. Returns False if task_id already exists."""
        with self._lock:
            if task.task_id in self._task_ids:
                return False
            heapq.heappush(self._heap, task)
            self._task_ids.add(task.task_id)
            return True

    def get_next(self) -> Optional[Task]:
        """Get the highest-priority pending task."""
        with self._lock:
            while self._heap:
                task = heapq.heappop(self._heap)
                if task.task_id in self._task_ids:
                    return task
            return None

    def complete(self, task_id: str, result: Any = None) -> bool:
        """Mark a task as completed."""
        with self._lock:
            if task_id not in self._task_ids:
                return False
            self._task_ids.discard(task_id)
            # Find and mark in heap (lazy deletion approach)
            for task in self._heap:
                if task.task_id == task_id:
                    task.result = result
                    self._completed[task_id] = task
                    break
            return True

    def cancel(self, task_id: str) -> bool:
        """Cancel a pending task."""
        with self._lock:
            if task_id not in self._task_ids:
                return False
            self._task_ids.discard(task_id)
            return True

    @property
    def pending_count(self) -> int:
        with self._lock:
            return len(self._task_ids)

    @property
    def tasks(self) -> list[Task]:
        with self._lock:
            return [t for t in self._heap if t.task_id in self._task_ids]
