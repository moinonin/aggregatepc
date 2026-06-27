# AggregatePC

**Distributed heterogeneous compute for idle PCs.**

Pool your spare computers — old laptops, desktops, Raspberry Pis — into a local cluster for distributed LLM inference and parallel computing. When your machines are idle, they contribute resources. When you need them back, they step aside automatically.

---

## Table of Contents

- [Why AggregatePC?](#whypc)
- [Quick Start](#quick-start)
- [Installation](#installation)
- [Configuration](#configuration)
- [Orchestration](#orchestration)
- [CLI Reference](#cli-reference)
- [Makefile Targets](#makefile-targets)
- [Architecture](#architecture)
- [Hardware Detection](#hardware-detection)
- [Idle Detection](#idle-detection)
- [Task Scheduling](#task-scheduling)
- [Network Discovery](#network-discovery)
- [Heartbeat & Health Monitoring](#heartbeat--health-monitoring)
- [Development](#development)
- [Roadmap](#roadmap)
- [Troubleshooting](#troubleshooting)

---

## Why AggregatePC?

You probably have several computers sitting around: an old laptop, a desktop you only use on weekends, a work machine that's idle in the evenings. AggregatePC turns these into a unified compute cluster:

- **Zero configuration** — Auto-detects hardware, no manual spec writing
- **Idle-only by default** — Workers monitor CPU/memory and step aside when you're using your machine
- **Cross-platform** — Windows, Linux, and macOS all work together
- **No cloud required** — Everything runs on your local network
- **VRAM-aware scheduling** — Knows which GPUs can hold which models and routes accordingly

---

## Quick Start

### 1. Clone and setup

```bash
git clone https://github.com/moinonin/aggregatepc.git
cd aggregatepc
python3 -m venv .venv
source .venv/bin/activate
pip install pytest  # optional, for running tests
```

### 2. Install Ollama and pull a model

```bash
# Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# Pull a model (choose based on your GPU/RAM)
ollama pull phi3:mini    # ~2.3GB, good for testing
ollama pull llama3:8b    # ~4.7GB, good for 8GB+ VRAM
ollama pull mistral:7b   # ~4.1GB, good alternative
```

### 3. Configure your cluster

Edit `configs/cluster.conf` with your machines' local IP addresses:

```ini
[controller]
192.168.1.5

[worker]
192.168.1.10
192.168.1.11
```

### 4. Start the controller (on one machine)

```bash
make controller
# or:
python3 aggregatepc.py controller
```

You'll see output like:

```
[aggregatepc] Starting controller on port 8765...
[aggregatepc] Controller IP: 192.168.1.5
[aggregatepc] Config has 2 worker(s): 192.168.1.10, 192.168.1.11
[aggregatepc] Workers can join with: aggregatepc worker
```

### 5. Join workers (on the other machines)

```bash
make worker
# or:
python3 aggregatepc.py worker
```

Workers read the controller IP from `configs/cluster.conf` automatically. No flags needed.

### 6. Check cluster status

```bash
make status
# or:
python3 aggregatepc.py status
```

Output:

```json
{
  "workers": [
    {
      "node_id": "office-desktop",
      "role": "worker",
      "status": "idle",
      "address": "192.168.1.10",
      "hardware": {
        "hostname": "office-desktop",
        "cpu_name": "Intel i7-12700K",
        "cpu_cores": 20,
        "ram_mb": 32768,
        "gpus": [
          {"name": "RTX 3060", "vram_mb": 12288, "integrated": false}
        ]
      },
      "compute_score": 376.0,
      "models": ["llama3:8b", "mistral:7b"]
    }
  ],
  "worker_count": 1,
  "available_count": 1,
  "cluster_metrics": {
    "total_cpu_cores": 20,
    "total_ram_mb": 32768,
    "total_ram_gb": 32.0,
    "total_vram_mb": 12288,
    "total_vram_gb": 12.0,
    "total_models": 2,
    "available_models": ["llama3:8b", "mistral:7b"]
  }
}
```

### 7. Start inference

Once workers have joined and Ollama is running:

```bash
make inference
```

This discovers the best available model across the cluster and starts serving it.

---

## Installation

### Requirements

- Python 3.10+
- Windows 10+, Linux (any modern distro), or macOS 12+
- All machines on the same local network (same subnet)
- **Ollama** (required for LLM inference) — https://ollama.com/download
- No other external dependencies

### Install from source (recommended)

```bash
git clone https://github.com/moinonin/aggregatepc.git
cd aggregatepc
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

This installs the `aggregatepc` command and the `zeroconf` dependency for mDNS auto-discovery.

### Install Ollama (required for inference)

```bash
# macOS/Linux:
curl -fsSL https://ollama.com/install.sh | sh

# Or with Homebrew:
brew install ollama

# Windows: download from https://ollama.com/download
```

Verify Ollama is working:
```bash
ollama version
```

### Install without packaging

If you don't want to install, you can run directly:

```bash
git clone https://github.com/moinonin/aggregatepc.git
cd aggregatepc
python3 -m venv .venv
source .venv/bin/activate
```

Then use `python3 aggregatepc.py` instead of `aggregatepc`.

### Install on worker machines only

Workers don't need `zeroconf` if you're providing the controller IP explicitly:

```bash
git clone https://github.com/moinonin/aggregatepc.git
cd aggregatepc
python3 -m venv .venv
source .venv/bin/activate
```

Workers can then join with:

```bash
python3 aggregatepc.py worker --controller <CONTROLLER_IP>
```

### Development install (includes test dependencies)

```bash
pip install -e ".[dev]"
# or manually:
pip install pytest zeroconf
```

### Verify installation

```bash
aggregatepc --help
# or without install:
python3 aggregatepc.py --help
```

You should see:

```
usage: aggregatepc [-h] [--config CONFIG]
                   {controller,worker,profile,status} ...

AggregatePC - Distributed heterogeneous compute for idle PCs
```

### Platform-specific notes

**Linux (Ubuntu/Debian):**
```bash
sudo apt install python3 python3-venv
```

**macOS:**
```bash
# Python 3.10+ is pre-installed on macOS 12+
# If not, use Homebrew:
brew install python@3.11
brew install ollama
```

**Windows:**
```powershell
# Use the official Python installer or Windows Store
# Then in PowerShell:
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e .
```

---

## Configuration

Configuration lives in `configs/cluster.conf`. The orchestration layer reads this file on startup.

### Format

```ini
# configs/cluster.conf
# One IP per line. Lines starting with # are comments.

[controller]
# Only one controller coordinates the cluster
192.168.1.5

[worker]
# One or more machines contributing idle compute
192.168.1.10
192.168.1.11
192.168.1.12

[ports]
# Optional: override default ports
# controller_port = 8765
# status_port = 8766
```

### Resolution Order

For any setting, AggregatePC uses this priority:

1. **CLI flags** — `aggregatepc worker --controller 10.0.0.1`
2. **Config file** — `configs/cluster.conf`
3. **Auto-discovery** — mDNS scan on the local network
4. **Defaults** — `127.0.0.1`, port `8765`

### Custom Config Path

```bash
aggregatepc worker --config /path/to/my-cluster.conf
```

---

## Orchestration

The orchestration flow runs from the controller terminal once workers have joined:

```
┌──────────────────────────────────────────────────────────┐
│                    CONTROLLER MACHINE                      │
│                                                           │
│  $ make controller                                        │
│  [aggregatepc] Starting controller on port 8765...        │
│  [aggregatepc] Controller IP: 192.168.1.5                 │
│  [aggregatepc] Config has 2 worker(s)                     │
│  [aggregatepc] Workers can join with: aggregatepc worker  │
│                                                           │
│  ← Workers connect via UDP (heartbeat on :8765)           │
│  ← Web dashboard available at :8766 (if enabled)          │
│  ← Tasks submitted here are dispatched to workers         │
└──────────────────────────────────────────────────────────┘
                            │
              ┌─────────────┼─────────────┐
              ▼             ▼             ▼
     ┌──────────────┐ ┌──────────────┐ ┌──────────────┐
     │   Worker 1   │ │   Worker 2   │ │   Worker 3   │
     │ 192.168.1.10 │ │ 192.168.1.11 │ │ 192.168.1.12 │
     │              │ │              │ │              │
     │ Idle? ✓ →    │ │ Idle? ✓ →    │ │ Idle? ✗ →    │
     │ Accept task  │ │ Accept task  │ │ Skip task    │
     └──────────────┘ └──────────────┘ └──────────────┘
```

### What happens at runtime

1. **Controller starts** — Reads `configs/cluster.conf`, prints worker IPs it expects
2. **Workers start** — Read controller IP from config, send UDP join request
3. **Controller registers workers** — Accepts join, starts tracking health
4. **Workers monitor idle state** — Check CPU/memory every 5 seconds
5. **Workers send heartbeats** — Every 10 seconds to controller
6. **Controller prunes stale workers** — No heartbeat in 60s = removed
7. **Tasks are scheduled** — Controller assigns to best-fit idle worker

### From the controller terminal

Once `make controller` is running, you see real-time logs:

```
[aggregatepc] Registered worker office-desktop (office-desktop)
[aggregatepc] Registered worker living-room-laptop (living-room-laptop)
[aggregatepc] Pruned 1 dead worker(s): old-laptop
```

Workers can join and leave at any time. The controller handles this automatically.

---

## CLI Reference

### `aggregatepc controller`

Start this machine as the cluster controller. Only one controller per cluster.

```bash
aggregatepc controller
aggregatepc controller --port 9000
```

| Flag | Default | Description |
|------|---------|-------------|
| `--port` | 8765 (or from config) | UDP port for worker connections |
| `--config` | `configs/cluster.conf` | Path to config file |

### `aggregatepc worker`

Start this machine as a worker. Joins the controller automatically.

```bash
aggregatepc worker
aggregatepc worker --controller 192.168.1.5
aggregatepc worker --controller 192.168.1.5 --cpu-threshold 15 --no-idle-check
```

| Flag | Default | Description |
|------|---------|-------------|
| `--controller` | from config / auto-discover | Controller IP address |
| `--port` | 8765 (or from config) | Controller port |
| `--cpu-threshold` | 25.0 | Max CPU % to be considered idle |
| `--mem-threshold` | 75.0 | Max memory % to be considered idle |
| `--idle-duration` | 30.0 | Seconds machine must be idle before accepting work |
| `--scan-timeout` | 3.0 | Seconds to scan for controller via mDNS |
| `--no-idle-check` | false | Accept work even when machine is in use |
| `--config` | `configs/cluster.conf` | Path to config file |

### `aggregatepc profile`

Detect local hardware and optionally scan the network for a cluster.

```bash
aggregatepc profile
aggregatepc profile --scan
aggregatepc profile --scan --output my-hardware.json
```

| Flag | Default | Description |
|------|---------|-------------|
| `--scan` | false | Also scan network for cluster |
| `--output` | stdout | Save profile to JSON file |
| `--scan-timeout` | 3.0 | Network scan timeout |
| `--config` | `configs/cluster.conf` | Path to config file |

### `aggregatepc status`

Query cluster status from the controller.

```bash
aggregatepc status
aggregatepc status --controller 192.168.1.5
aggregatepc status --controller 192.168.1.5 --port 9000
```

| Flag | Default | Description |
|------|---------|-------------|
| `--controller` | from config / 127.0.0.1 | Controller IP to query |
| `--port` | 8766 (or from config) | Controller status port |
| `--config` | `configs/cluster.conf` | Path to config file |

---

## Makefile Targets

```bash
make help                 # Show all available targets with descriptions
make controller           # Start as cluster controller
make worker               # Start as worker (auto-discover or from config)
make worker CONTROLLER=192.168.1.5  # Join specific controller
make profile              # Profile local hardware
make profile SCAN=1       # Profile + network scan
make status               # Show cluster status
make status CONTROLLER=192.168.1.5  # Query specific controller
make inference            # Start inference with best available model
make test                 # Run test suite (pytest)
make clean                # Remove __pycache__ and .pyc files
make install              # Attempt pip install -e . (if pyproject.toml exists)
```

### Passing variables

```bash
# Override controller IP for worker join
make worker CONTROLLER=10.0.0.5

# Override controller for status query
make status CONTROLLER=10.0.0.5 PORT=9000

# Profile with network scan
make scan
```

---

## Architecture

```
aggregatepc.py              ← CLI entry point (argparse)
    │
    ├── cluster/
    │   ├── config.py        ← Config loader (configs/cluster.conf)
    │   ├── detect.py        ← Cross-platform hardware detection
    │   ├── nodes/
    │   │   ├── __init__.py  ← Node abstraction (roles, status, scoring)
    │   │   ├── worker.py    ← Worker daemon (idle detection, heartbeat)
    │   │   └── controller.py ← Controller (worker registry, health monitor)
    │   ├── network/
    │   │   ├── discovery.py ← mDNS peer discovery + UDP broadcast fallback
    │   │   └── heartbeat.py ← UDP heartbeat listener + status query handler
    │   └── compute/
    │       ├── task_queue.py    ← Thread-safe priority queue
    │       ├── gpu_allocator.py ← VRAM-aware model placement
    │       └── scheduler.py     ← Capability-based task assignment
    │
    ├── configs/
    │   └── cluster.conf     ← IP addresses and ports
    │
    ├── scripts/
    │   ├── auto_profile.py  ← Standalone hardware profiler
    │   └── start_worker.py  ← Legacy worker startup script
    │
    └── tasks/
        ├── llm_inference.py  ← LLM inference task type
        └── batch_compute.py  ← General batch processing task type
```

---

## Hardware Detection

`cluster/detect.py` auto-detects hardware on Windows, Linux, and macOS:

```python
from cluster.detect import detect_hardware

profile = detect_hardware()
print(f"CPU: {profile.cpu.name} ({profile.cpu.cores_logical} cores)")
print(f"RAM: {profile.memory.total_mb} MB")
for gpu in profile.gpus:
    print(f"GPU: {gpu.name} ({gpu.vram_mb} MB VRAM)")
```

**Platform-specific detection:**

| Platform | CPU | GPU | Memory |
|----------|-----|-----|--------|
| Windows | WMI (`Win32_Processor`) | WMI (`Win32_VideoController`) | WMI (`Win32_PhysicalMemory`) |
| Linux | `lscpu` | `nvidia-smi`, `lspci` | `/proc/meminfo` |
| macOS | `sysctl` | `system_profiler SPDisplaysDataType` | `sysctl hw.memsize` |

---

## Idle Detection

Workers continuously monitor system usage to avoid interfering with your work:

```python
from cluster.nodes.worker import IdleThreshold

# Conservative: only work when PC is truly idle
threshold = IdleThreshold(
    cpu_percent_max=10.0,      # CPU must be under 10%
    memory_percent_max=50.0,   # Memory must be under 50%
    idle_duration_seconds=60.0 # Must be idle for 60 seconds
)

# Aggressive: contribute even under light load
threshold = IdleThreshold(
    cpu_percent_max=50.0,
    memory_percent_max=85.0,
    idle_duration_seconds=10.0
)
```

**State machine:**

```
BUSY ──(cpu < threshold AND mem < threshold)──→ entering idle timer
  ↑                                                    │
  │                                              (timer running)
  │                                                    │
  └──(user becomes active)──────────────────── IDLE (available for tasks)
```

---

## Task Scheduling

The scheduler assigns tasks to the best-suited node:

```python
from cluster.compute.scheduler import TaskScheduler, Task, Priority

scheduler = TaskScheduler()
scheduler.register_node(worker_node_1)
scheduler.register_node(worker_node_2)

task = Task(
    task_id="infer-001",
    task_type="llm_inference",
    priority=Priority.HIGH,
    required_vram_mb=8000,    # Needs 8GB GPU
    required_ram_mb=16384,    # Needs 16GB system RAM
    required_cpu_cores=4,
    payload={"model": "llama-7b", "prompt": "Explain quantum computing"},
)

scheduler.submit_task(task)
assignment = scheduler.schedule_next()
print(f"Assigned to: {assignment.node_id}")
```

**Scoring algorithm:**

1. Filter out nodes that can't meet requirements (insufficient VRAM/RAM/cores)
2. Score remaining nodes:
   - VRAM fit ratio (tighter fit = higher score, avoids waste)
   - Available CPU cores
   - Idle nodes preferred over busy ones
3. Assign to highest-scoring node

---

## Network Discovery

Workers can find the controller automatically via mDNS:

```python
from cluster.network.discovery import discover_peers

peers = discover_peers(timeout=3.0)
if peers:
    print(f"Found controller at {peers[0].address}")
```

**Priority:**

1. If `--controller` flag is provided, use that IP
2. If `configs/cluster.conf` has a controller IP, use that
3. Scan the local network via mDNS (requires `zeroconf` package)
4. Fall back to UDP broadcast discovery
5. Give up with helpful error message

Install `zeroconf` for best auto-discovery:

```bash
pip install zeroconf
```

---

## Heartbeat & Health Monitoring

Workers send heartbeats every 10 seconds (configurable). The controller tracks:

| State | Condition | Action |
|-------|-----------|--------|
| IDLE | Heartbeat received, CPU/mem low | Available for tasks |
| BUSY | Heartbeat received, CPU/mem high | Skip task assignment |
| STALE | No heartbeat for 30s | Mark unresponsive |
| OFFLINE | No heartbeat for 60s | Remove from cluster |

```python
from cluster.network.heartbeat import HeartbeatMonitor

monitor = HeartbeatMonitor()
monitor.register(worker_node)

# Check health
print(f"Workers: {monitor.worker_count}, Available: {monitor.available_count}")

# Prune dead workers
dead = monitor.prune_dead()
if dead:
    print(f"Removed: {dead}")
```

---

## Development

### Running tests

```bash
make test
# or:
python3 -m pytest tests/ -v
```

### Project structure

```
.
├── aggregatepc.py          # CLI entry point
├── Makefile                # Build targets
├── README.md               # This file
├── cluster/
│   ├── __init__.py         # Package version
│   ├── config.py           # Config file loader
│   ├── detect.py           # Hardware detection
│   ├── nodes/              # Node abstraction, worker, controller
│   ├── network/            # Discovery, heartbeat
│   └── compute/            # Task queue, GPU allocator, scheduler
├── configs/
│   └── cluster.conf        # Your cluster IPs go here
├── scripts/
│   ├── auto_profile.py     # Standalone profiler
│   └── start_worker.py     # Legacy worker script
├── tasks/
│   ├── llm_inference.py    # LLM task type
│   └── batch_compute.py    # Batch task type
├── tests/
│   ├── __init__.py
│   └── test_basic.py       # 28 tests covering all modules
└── SPEC.md                 # Cluster configuration specification
```

### Adding a new task type

```python
# tasks/my_task.py
from cluster.compute.task_queue import Task, Priority

def create_my_task(task_id: str, params: dict) -> Task:
    return Task(
        task_id=task_id,
        task_type="my_task",
        priority=Priority.NORMAL,
        required_ram_mb=params.get("ram_mb", 1024),
        required_vram_mb=params.get("vram_mb", 0),
        required_cpu_cores=params.get("cpu_cores", 1),
        payload=params,
    )
```

---

## Roadmap

- [x] Sprint 1: Hardware auto-detection (Windows/Linux/macOS)
- [x] Sprint 2: Cluster formation, worker join, heartbeat monitoring
- [x] Sprint 3: Task scheduling, VRAM-aware allocation, priority queue
- [x] Config file support, unified CLI, Makefile
- [ ] Sprint 4: Web dashboard, contribution stats, GUI
- [ ] Sprint 5: Encrypted communication, auto-updates, pip packaging

---

## Troubleshooting

### Worker can't find controller

```bash
# Check config file
cat configs/cluster.conf

# Try explicit IP
aggregatepc worker --controller 192.168.1.5

# Check firewall (Windows)
# Open PowerShell as Admin:
Set-NetConnectionProfile -InterfaceAlias (Get-NetAdapter | Where-Object Status -eq "Up").Name -NetworkCategory Private
```

### No workers showing in status

1. Verify all machines are on the same network/subnet
2. Check that the controller is running before workers start
3. Look at worker output for "Joined!" message
4. If using mDNS: `pip install zeroconf` on all machines

### Port already in use

```bash
# Use a different port
aggregatepc controller --port 9000
aggregatepc worker --controller 192.168.1.5 --port 9000
```

Or set in config:

```ini
[ports]
controller_port = 9000
status_port = 9001
```

---

## License

MIT
