# Project Sprints & Implementation Plan

**Project Goal:** Build an automated GPU/CPU aggregation system that enables users with free PCs to pool their idle computing resources for distributed LLM inference and general-purpose parallel computing through a simple, user-friendly interface.

---

## Sprint 1: Core Infrastructure & Auto-Detection

### Objective
Enable automatic detection and profiling of local hardware resources without manual configuration.

### Tasks
- [ ] `cluster/detect.py` - Auto-detect CPU cores, RAM, GPU models, and VRAM across platforms
- [ ] `cluster/network/discovery.py` - Automatic local network discovery using mDNS/SSDP
- [ ] `scripts/auto_profile.py` - One-click hardware and network profiling script
- [ ] `cluster/nodes/__init__.py` - Node abstraction layer (Windows, Linux, macOS)

### Platform-Specific Handling
- Windows: WMI queries for CPU/GPU, psutil for memory
- Linux: lscpu, nvidia-smi, /proc/meminfo parsing
- macOS: system_profiler, unified memory reporting

---

## Sprint 2: Cluster Formation & Bootstrap

### Objective
Create a zero-config cluster formation process where users can claim "worker" status on their machines.

### Tasks
- [ ] `cluster/nodes/worker.py` - Lightweight worker daemon for idle PCs
- [ ] `cluster/nodes/controller.py` - Controller that aggregates workers
- [ ] `scripts/start_worker.py` - Simple script to turn any PC into a worker
- [ ] `cluster/network/heartbeat.py` - Worker health monitoring and auto-reconnect
- [ ] CLI: `aggregatepc worker --idle-threshold=10%` - Auto-start when PC is idle

### User Story
> Users run `./start_worker.sh` on their old laptops/desktops. The script auto-detects hardware, joins the local network cluster, and waits for tasks.

---

## Sprint 3: Task Distribution & Load Balancing

### Objective
Intelligently distribute compute tasks across heterogeneous nodes based on their capabilities.

### Tasks
- [ ] `cluster/compute/scheduler.py` - Capability-based task assignment
- [ ] `cluster/compute/task_queue.py` - Priority queue with resource requirements
- [ ] `cluster/compute/gpu_allocator.py` - VRAM-aware model placement
- [ ] `cluster/tasks/llm_inference.py` - LLM inference task type
- [ ] `cluster/tasks/batch_compute.py` - General batch processing task type

### Key Features
- Auto-split large models across nodes with available VRAM
- Fall back to CPU when GPU memory insufficient
- Task priority based on urgency and node availability

---

## Sprint 4: User-Friendly Control Interface

### Objective
Provide simple interfaces for non-technical users to contribute their PC's idle resources.

### Tasks
- [ ] `cluster/cli/__init__.py` - Unified CLI with intuitive commands
- [ ] `cluster/gui/app.py` - Optional simple GUI showing contribution stats
- [ ] `cluster/contrib/dashboard.py` - Web dashboard showing cluster status
- [ ] `scripts/aggregatepc.py` - Main entry point: `python aggregatepc.py start`

### CLI Examples
```bash
# Join your PC to the cluster
aggregatepc worker start

# See what your PC contributed this month
aggregatepc stats

# Temporarily pause when you need your PC
aggregatepc worker pause
```

---

## Sprint 5: Production Hardening & Community Release

### Objective
Make the system stable for everyday use and easy for others to set up.

### Tasks
- [ ] Cross-platform installer script
- [ ] Auto-update mechanism for workers
- [ ] Security: encrypted task communication, token authentication
- [ ] Documentation: "Set up your old laptop in 5 minutes or less"
- [ ] Example configs for common use cases (LLM inference, video encoding, etc.)

---

## Implementation Principles

1. **Users First, Developers Second** - Every feature must pass the "grandma test" (easy enough for a non-developer to use)
2. **Idle-Only Policy** - Workers should never interfere when the user is actively using their PC
3. **Any PC Works** - Support Windows 10+, Linux, macOS with any hardware configuration
4. **No Cloud Required** - Everything runs on local network, no external dependencies
5. **Plug & Play** - Auto-discovery and zero-config setup whenever possible