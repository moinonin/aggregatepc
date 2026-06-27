"""Hardware detection for CPU, RAM, and GPU across Windows, Linux, and macOS."""

from __future__ import annotations

import platform
import subprocess
import os
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class GPUInfo:
    name: str
    vram_mb: int
    is_integrated: bool = False
    vendor: str = ""  # "nvidia", "amd", "apple", "intel"


@dataclass
class CPUInfo:
    name: str
    cores_physical: int
    cores_logical: int
    architecture: str  # "x86_64", "arm64", etc.


@dataclass
class MemoryInfo:
    total_mb: int
    available_mb: int


@dataclass
class HardwareProfile:
    cpu: CPUInfo
    memory: MemoryInfo
    gpus: list[GPUInfo] = field(default_factory=list)
    hostname: str = ""
    os_name: str = ""
    os_version: str = ""

    @property
    def total_vram_mb(self) -> int:
        return sum(g.vram_mb for g in self.gpus)

    @property
    def has_discrete_gpu(self) -> bool:
        return any(not g.is_integrated for g in self.gpus)

    @property
    def primary_gpu(self) -> Optional[GPUInfo]:
        if not self.gpus:
            return None
        # Prefer discrete over integrated
        discrete = [g for g in self.gpus if not g.is_integrated]
        return discrete[0] if discrete else self.gpus[0]


def _run(cmd: list[str], timeout: int = 10) -> str:
    """Run a command and return stdout, or empty string on failure."""
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return result.stdout.strip()
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return ""


def detect_cpu() -> CPUInfo:
    """Detect CPU information for the current platform."""
    system = platform.system()
    cpu_name = platform.processor() or "Unknown"
    arch = platform.machine()
    detected_cores = os.cpu_count() or 1
    cores_physical = detected_cores
    cores_logical = detected_cores

    if system == "Windows":
        out = _run(["wmic", "cpu", "get", "Name,NumberOfCores,NumberOfLogicalProcessors", "/format:csv"])
        for line in out.splitlines():
            parts = line.split(",")
            if len(parts) >= 4 and parts[1].strip():
                cpu_name = parts[1].strip()
                try:
                    cores_physical = int(parts[2].strip())
                    cores_logical = int(parts[3].strip())
                except ValueError:
                    pass

    elif system == "Linux":
        lscpu = _run(["lscpu"])
        for line in lscpu.splitlines():
            if "Model name:" in line:
                cpu_name = line.split(":", 1)[1].strip()
            if "CPU(s):" in line and "per" not in line and "NUMA" not in line:
                try:
                    cores_logical = int(line.split(":", 1)[1].strip())
                except ValueError:
                    pass
        cores_physical = cores_logical  # Simplified; could parse Core(s) per socket

    elif system == "Darwin":
        out = _run(["sysctl", "-n", "machdep.cpu.brand_string"])
        if out:
            cpu_name = out
        out = _run(["sysctl", "-n", "hw.physicalcpu"])
        if out:
            try:
                cores_physical = int(out)
            except ValueError:
                cores_physical = 1
        out = _run(["sysctl", "-n", "hw.logicalcpu"])
        if out:
            try:
                cores_logical = int(out)
            except ValueError:
                cores_logical = 1

    return CPUInfo(
        name=cpu_name,
        cores_physical=cores_physical,
        cores_logical=cores_logical,
        architecture=arch,
    )


def detect_memory() -> MemoryInfo:
    """Detect system memory for the current platform."""
    system = platform.system()
    total_mb = 0
    available_mb = 0

    if system == "Windows":
        out = _run(["wmic", "os", "get", "TotalVisibleMemorySize,FreePhysicalMemory", "/format:csv"])
        for line in out.splitlines():
            parts = line.split(",")
            if len(parts) >= 3 and parts[1].strip():
                try:
                    total_mb = int(parts[1].strip()) // 1024  # KB to MB
                    available_mb = int(parts[2].strip()) // 1024
                except ValueError:
                    pass

    elif system == "Linux":
        try:
            with open("/proc/meminfo", "r") as f:
                meminfo = f.read()
        except OSError:
            meminfo = ""
        for line in meminfo.splitlines():
            if line.startswith("MemTotal:"):
                try:
                    total_mb = int(line.split()[1]) // 1024
                except (ValueError, IndexError):
                    pass
            elif line.startswith("MemAvailable:"):
                try:
                    available_mb = int(line.split()[1]) // 1024
                except (ValueError, IndexError):
                    pass

    elif system == "Darwin":
        out = _run(["sysctl", "-n", "hw.memsize"])
        if out:
            try:
                total_mb = int(out) // (1024 * 1024)
            except ValueError:
                pass
        # Approximate available memory from vm_stat
        vm_stat = _run(["vm_stat"])
        page_size = 4096
        free_pages = 0
        for line in vm_stat.splitlines():
            if "page size of" in line:
                try:
                    page_size = int(line.split()[-1])
                except (ValueError, IndexError):
                    pass
            elif "Pages free:" in line:
                try:
                    free_pages = int(line.split()[-1].rstrip("."))
                except (ValueError, IndexError):
                    pass
        available_mb = (free_pages * page_size) // (1024 * 1024)

    if total_mb <= 0:
        total_mb = _detect_total_memory_mb()
    if available_mb <= 0:
        available_mb = total_mb

    return MemoryInfo(total_mb=total_mb, available_mb=available_mb)


def _detect_total_memory_mb() -> int:
    """Best-effort total system memory fallback using POSIX sysconf."""
    try:
        page_size = os.sysconf("SC_PAGE_SIZE")
        page_count = os.sysconf("SC_PHYS_PAGES")
        return max((page_size * page_count) // (1024 * 1024), 1)
    except (AttributeError, OSError, ValueError):
        return 1


def detect_gpus() -> list[GPUInfo]:
    """Detect GPUs for the current platform."""
    system = platform.system()
    gpus: list[GPUInfo] = []

    if system == "Windows":
        out = _run(["wmic", "path", "win32_VideoController", "get", "Name,AdapterRAM", "/format:csv"])
        for line in out.splitlines():
            parts = line.split(",")
            if len(parts) >= 3 and parts[1].strip():
                name = parts[1].strip()
                try:
                    vram = int(parts[2].strip())
                except ValueError:
                    vram = 0
                is_integrated = any(kw in name.lower() for kw in ["intel", "uhd", "iris"])
                vendor = "intel" if is_integrated else ("nvidia" if "nvidia" in name.lower() else "amd")
                gpus.append(GPUInfo(
                    name=name,
                    vram_mb=vram // (1024 * 1024) if vram > 0 else 0,
                    is_integrated=is_integrated,
                    vendor=vendor,
                ))

    elif system == "Linux":
        out = _run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"])
        if out:
            for line in out.splitlines():
                parts = line.split(",")
                if len(parts) >= 2:
                    name = parts[0].strip()
                    vram_str = parts[1].strip().replace("MiB", "").replace("MB", "").strip()
                    try:
                        vram = int(vram_str)
                    except ValueError:
                        vram = 0
                    gpus.append(GPUInfo(name=name, vram_mb=vram, is_integrated=False, vendor="nvidia"))
        # Check for integrated GPUs via lspci
        lspci = _run(["lspci"])
        for line in lspci.splitlines():
            if "VGA" in line or "3D" in line:
                name = line.split(":", 2)[-1].strip() if ":" in line else line
                if not any(g.name in line for g in gpus):
                    is_integrated = any(kw in name.lower() for kw in ["intel", "amd", "ati"])
                    vendor = "intel" if "intel" in name.lower() else ("amd" if "amd" in name.lower() or "ati" in name.lower() else "unknown")
                    gpus.append(GPUInfo(name=name, vram_mb=0, is_integrated=is_integrated, vendor=vendor))

    elif system == "Darwin":
        # Apple Silicon: unified memory, no dedicated VRAM
        profiler = _run(["system_profiler", "SPDisplaysDataType"])
        for block in profiler.split("\n\n"):
            if "Chipset Model" in block:
                name = ""
                vram = 0
                for line in block.splitlines():
                    if "Chipset Model:" in line:
                        name = line.split(":", 1)[1].strip()
                    elif "VRAM" in line:
                        vram_str = line.split(":", 1)[1].strip().replace("GB", "").replace("MB", "").strip()
                        try:
                            vram = int(vram_str)
                            if "GB" in line:
                                vram *= 1024
                        except ValueError:
                            vram = 0
                if name:
                    gpus.append(GPUInfo(
                        name=name,
                        vram_mb=vram,
                        is_integrated=True,
                        vendor="apple",
                    ))

    return gpus


def detect_hardware() -> HardwareProfile:
    """Detect full hardware profile for the current machine."""
    return HardwareProfile(
        cpu=detect_cpu(),
        memory=detect_memory(),
        gpus=detect_gpus(),
        hostname=platform.node(),
        os_name=platform.system(),
        os_version=platform.version(),
    )


if __name__ == "__main__":
    import json

    profile = detect_hardware()
    print(json.dumps({
        "hostname": profile.hostname,
        "os": f"{profile.os_name} {profile.os_version}",
        "cpu": {
            "name": profile.cpu.name,
            "physical_cores": profile.cpu.cores_physical,
            "logical_cores": profile.cpu.cores_logical,
            "architecture": profile.cpu.architecture,
        },
        "memory": {
            "total_mb": profile.memory.total_mb,
            "available_mb": profile.memory.available_mb,
        },
        "gpus": [
            {
                "name": g.name,
                "vram_mb": g.vram_mb,
                "integrated": g.is_integrated,
                "vendor": g.vendor,
            }
            for g in profile.gpus
        ],
    }, indent=2))
