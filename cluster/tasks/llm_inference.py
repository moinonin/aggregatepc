"""LLM inference task definition."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from cluster.compute.task_queue import Priority, Task


@dataclass
class LLMInferenceRequest:
    """Parameters for an LLM inference task."""
    model_name: str
    prompt: str
    max_tokens: int = 256
    temperature: float = 0.7
    top_p: float = 0.9
    quantization: str = "fp16"  # "fp16", "int8", "int4"
    supports_cpu_fallback: bool = True

    def to_task(self, task_id: str, priority: Priority = Priority.NORMAL) -> Task:
        """Convert this request to a Task for the queue."""
        from cluster.compute.gpu_allocator import estimate_model_vram

        vram_needed = estimate_model_vram(self.model_name, self.quantization)
        # Estimate RAM for CPU fallback (roughly 2x model size in fp16)
        ram_needed = vram_needed * 2 if self.supports_cpu_fallback else vram_needed

        return Task(
            task_id=task_id,
            task_type="llm_inference",
            priority=priority,
            required_ram_mb=ram_needed,
            required_vram_mb=vram_needed,
            required_cpu_cores=4,
            payload={
                "model": self.model_name,
                "prompt": self.prompt,
                "max_tokens": self.max_tokens,
                "temperature": self.temperature,
                "top_p": self.top_p,
                "quantization": self.quantization,
            },
        )


@dataclass
class LLMInferenceResult:
    """Result of an LLM inference task."""
    task_id: str
    node_id: str
    output_text: str
    tokens_generated: int
    latency_ms: float
    model_name: str
    error: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "node_id": self.node_id,
            "output_text": self.output_text,
            "tokens_generated": self.tokens_generated,
            "latency_ms": self.latency_ms,
            "model_name": self.model_name,
            "error": self.error,
        }
