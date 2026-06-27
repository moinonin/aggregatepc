"""VRAM-aware GPU allocator for distributed model placement."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

from cluster.nodes import Node

logger = logging.getLogger("aggregatepc.gpu_allocator")


@dataclass
class ModelRequirements:
    """Resource requirements for running a model."""
    model_name: str
    vram_mb: int          # GPU memory needed
    ram_mb: int           # System RAM needed (for offloading)
    cpu_cores: int = 1    # CPU cores needed
    supports_cpu_fallback: bool = False  # Can run on CPU if no GPU
    quantization_levels: list[str] = None  # Supported: "fp16", "int8", "int4"

    def __post_init__(self):
        if self.quantization_levels is None:
            self.quantization_levels = ["fp16", "int8", "int4"]


@dataclass
class AllocationPlan:
    """How a model is placed across the cluster."""
    model_name: str
    fits_on_single_gpu: bool
    requires_split: bool
    allocations: list[dict]  # [{node_id, vram_mb, cpu_cores}]
    total_vram_mb: int
    total_ram_mb: int
    uses_cpu_fallback: bool

    @property
    def node_count(self) -> int:
        return len(self.allocations)


def estimate_model_vram(model_name: str, quantization: str = "fp16") -> int:
    """Estimate VRAM needed for a model based on name and quantization."""
    # Extract parameter count from model name (e.g., "llama-7b", "mistral-8x7b")
    name_lower = model_name.lower()
    params_b = 0

    # Try to parse parameter count from model name
    import re
    match = re.search(r"(\d+(?:\.\d+)?)[Bb]", name_lower)
    if match:
        params_b = float(match.group(1))

    # Rough VRAM estimation: bytes per param depends on quantization
    bytes_per_param = {"fp16": 2, "int8": 1, "int4": 0.5, "fp32": 4}
    bpq = bytes_per_param.get(quantization, 2)

    # Add 20% overhead for KV cache and intermediate activations
    vram_mb = int(params_b * 1e9 * bpq / (1024 * 1024) * 1.2)
    return max(vram_mb, 512)  # Minimum 512MB


class GPUAllocator:
    """Allocates models across cluster nodes based on available resources."""

    def __init__(self, nodes: list[Node] = None):
        self._nodes = nodes or []

    def update_nodes(self, nodes: list[Node]) -> None:
        """Update the set of available nodes."""
        self._nodes = [n for n in nodes if n.is_available]

    def can_fit_model(self, model: ModelRequirements) -> bool:
        """Check if the cluster can fit a model at any quantization level."""
        for quant in model.quantization_levels:
            vram_needed = estimate_model_vram(model.model_name, quant)
            for node in self._nodes:
                available_vram = sum(
                    g.vram_mb for g in node.hardware.gpus
                )
                if available_vram >= vram_needed:
                    return True

        # Check CPU fallback
        if model.supports_cpu_fallback:
            for node in self._nodes:
                if node.hardware.memory.total_mb >= model.ram_mb:
                    return True

        return False

    def allocate(self, model: ModelRequirements) -> Optional[AllocationPlan]:
        """Find the best allocation for a model across available nodes."""
        # Try each quantization level from highest quality to lowest
        for quant in model.quantization_levels:
            vram_needed = estimate_model_vram(model.model_name, quant)

            # First try: single GPU that can hold the model
            for node in self._nodes:
                available_vram = sum(g.vram_mb for g in node.hardware.gpus)
                if available_vram >= vram_needed:
                    return AllocationPlan(
                        model_name=model.model_name,
                        fits_on_single_gpu=True,
                        requires_split=False,
                        allocations=[{
                            "node_id": node.node_id,
                            "vram_mb": vram_needed,
                            "cpu_cores": model.cpu_cores,
                            "quantization": quant,
                        }],
                        total_vram_mb=vram_needed,
                        total_ram_mb=model.ram_mb,
                        uses_cpu_fallback=False,
                    )

            # Second try: split across multiple GPUs
            total_available = sum(
                sum(g.vram_mb for g in n.hardware.gpus) for n in self._nodes
            )
            if total_available >= vram_needed:
                allocations = []
                remaining = vram_needed
                for node in self._nodes:
                    node_vram = sum(g.vram_mb for g in node.hardware.gpus)
                    if node_vram > 0 and remaining > 0:
                        allocated = min(node_vram, remaining)
                        allocations.append({
                            "node_id": node.node_id,
                            "vram_mb": allocated,
                            "cpu_cores": max(1, model.cpu_cores // len(self._nodes)),
                            "quantization": quant,
                        })
                        remaining -= allocated
                        if remaining <= 0:
                            break

                if remaining <= 0:
                    return AllocationPlan(
                        model_name=model.model_name,
                        fits_on_single_gpu=False,
                        requires_split=True,
                        allocations=allocations,
                        total_vram_mb=vram_needed,
                        total_ram_mb=model.ram_mb,
                        uses_cpu_fallback=False,
                    )

        # Last resort: CPU fallback
        if model.supports_cpu_fallback:
            for node in self._nodes:
                if node.hardware.memory.total_mb >= model.ram_mb:
                    return AllocationPlan(
                        model_name=model.model_name,
                        fits_on_single_gpu=False,
                        requires_split=False,
                        allocations=[{
                            "node_id": node.node_id,
                            "vram_mb": 0,
                            "cpu_cores": model.cpu_cores,
                            "quantization": "cpu",
                        }],
                        total_vram_mb=0,
                        total_ram_mb=model.ram_mb,
                        uses_cpu_fallback=True,
                    )

        return None

    def get_cluster_capacity(self) -> dict:
        """Get total cluster compute capacity."""
        total_vram = 0
        total_ram = 0
        total_cores = 0
        gpu_count = 0

        for node in self._nodes:
            total_ram += node.hardware.memory.total_mb
            total_cores += node.hardware.cpu.cores_logical
            for gpu in node.hardware.gpus:
                total_vram += gpu.vram_mb
                gpu_count += 1

        return {
            "total_vram_mb": total_vram,
            "total_ram_mb": total_ram,
            "total_cpu_cores": total_cores,
            "gpu_count": gpu_count,
            "node_count": len(self._nodes),
        }
