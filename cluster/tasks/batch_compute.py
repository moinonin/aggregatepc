"""General batch processing task definition."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from cluster.compute.task_queue import Priority, Task


@dataclass
class BatchComputeRequest:
    """Parameters for a batch compute task."""
    job_name: str
    command: str  # Shell command or Python function reference
    args: list[str] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)
    timeout_seconds: int = 300
    estimated_ram_mb: int = 1024
    estimated_cpu_cores: int = 2
    use_gpu: bool = False
    estimated_vram_mb: int = 0

    def to_task(self, task_id: str, priority: Priority = Priority.NORMAL) -> Task:
        """Convert this request to a Task for the queue."""
        return Task(
            task_id=task_id,
            task_type="batch_compute",
            priority=priority,
            required_ram_mb=self.estimated_ram_mb,
            required_vram_mb=self.estimated_vram_mb,
            required_cpu_cores=self.estimated_cpu_cores,
            payload={
                "job_name": self.job_name,
                "command": self.command,
                "args": self.args,
                "env": self.env,
                "timeout_seconds": self.timeout_seconds,
            },
        )


@dataclass
class BatchComputeResult:
    """Result of a batch compute task."""
    task_id: str
    node_id: str
    stdout: str
    stderr: str
    return_code: int
    duration_seconds: float
    error: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "node_id": self.node_id,
            "stdout": self.stdout,
            "stderr": self.stderr,
            "return_code": self.return_code,
            "duration_seconds": self.duration_seconds,
            "error": self.error,
        }


def run_batch_command(
    command: str,
    args: list[str],
    env: dict[str, str],
    timeout: int,
) -> BatchComputeResult:
    """Execute a batch command on a node. Called by worker nodes."""
    import subprocess
    import uuid

    start = time.time()
    try:
        full_env = {**__import__("os").environ, **env}
        result = subprocess.run(
            [command] + args,
            capture_output=True,
            text=True,
            timeout=timeout,
            env=full_env,
        )
        duration = time.time() - start
        return BatchComputeResult(
            task_id=str(uuid.uuid4()),
            node_id="",
            stdout=result.stdout,
            stderr=result.stderr,
            return_code=result.returncode,
            duration_seconds=duration,
        )
    except subprocess.TimeoutExpired:
        duration = time.time() - start
        return BatchComputeResult(
            task_id=str(uuid.uuid4()),
            node_id="",
            stdout="",
            stderr="Task timed out",
            return_code=-1,
            duration_seconds=duration,
            error=f"Timed out after {timeout}s",
        )
    except Exception as e:
        duration = time.time() - start
        return BatchComputeResult(
            task_id=str(uuid.uuid4()),
            node_id="",
            stdout="",
            stderr=str(e),
            return_code=-1,
            duration_seconds=duration,
            error=str(e),
        )
