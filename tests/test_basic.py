"""Tests for AggregatePC core functionality."""

import sys
import os
import time
import json
import subprocess

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cluster.detect import detect_hardware, detect_cpu, detect_memory
from cluster.compute.gpu_allocator import estimate_model_vram
from cluster.nodes import Node, NodeRole, NodeStatus, create_local_node
from cluster.nodes.worker import WorkerDaemon, WorkerConfig, IdleThreshold
from cluster.nodes.controller import ClusterController
from cluster.network.discovery import discover_peers, MDNSDiscovery
from cluster.network.heartbeat import HeartbeatListener, HeartbeatMonitor
from cluster.compute.task_queue import Task, TaskQueue, Priority
from cluster.compute.gpu_allocator import GPUAllocator, ModelRequirements, AllocationPlan
from cluster.compute.scheduler import TaskScheduler
from cluster.tasks.llm_inference import LLMInferenceRequest, LLMInferenceResult
from cluster.tasks.batch_compute import BatchComputeRequest, BatchComputeResult


# --- Test Fixtures ---

def make_node(node_id, cpu_cores=8, ram_mb=16384, vram_mb=12288, integrated=False):
    from cluster.detect import CPUInfo, MemoryInfo, GPUInfo, HardwareProfile
    gpu_info = [GPUInfo(name="Test GPU", vram_mb=vram_mb, is_integrated=integrated, vendor="nvidia")] if vram_mb > 0 else []
    return Node(
        node_id=node_id,
        role=NodeRole.WORKER,
        hardware=HardwareProfile(
            cpu=CPUInfo(name="Test CPU", cores_physical=cpu_cores, cores_logical=cpu_cores, architecture="x86_64"),
            memory=MemoryInfo(total_mb=ram_mb, available_mb=ram_mb // 2),
            gpus=gpu_info,
            hostname=node_id,
        ),
        status=NodeStatus.IDLE,
    )


# --- Sprint 1: Hardware Detection ---

class TestHardwareDetection:
    def test_detect_cpu(self):
        cpu = detect_cpu()
        assert cpu.cores_logical > 0
        assert cpu.name != ""

    def test_detect_memory(self):
        mem = detect_memory()
        assert mem.total_mb > 0

    def test_detect_hardware_full(self):
        profile = detect_hardware()
        assert profile.cpu.cores_logical > 0
        assert profile.memory.total_mb > 0
        assert profile.hostname != ""

    def test_create_local_node(self):
        node = create_local_node(NodeRole.WORKER)
        assert node.role == NodeRole.WORKER
        assert node.hardware.cpu.cores_logical > 0


# --- Sprint 1: Network Discovery ---

class TestNetworkDiscovery:
    def test_get_local_ip(self):
        disc = MDNSDiscovery(port=19999)
        ip = disc.get_local_ip()
        assert ip != ""
        assert "." in ip

    def test_discover_peers_returns_list(self):
        peers = discover_peers(timeout=1.0)
        assert isinstance(peers, list)


# --- Sprint 2: Cluster Formation ---

class TestClusterFormation:
    def test_worker_creation(self):
        config = WorkerConfig(worker=IdleThreshold(cpu_percent_max=25.0, idle_duration_seconds=30.0))
        daemon = WorkerDaemon(config=config)
        assert daemon.node.role == NodeRole.WORKER
        assert daemon.node.hardware.cpu.cores_logical > 0

    def test_worker_join_controller(self):
        """Full integration: worker joins controller via UDP."""
        controller = ClusterController(port=18900)
        controller._running = True
        controller._heartbeat_listener.start()

        config = WorkerConfig(controller_port=18900, worker=IdleThreshold(idle_duration_seconds=0))
        daemon = WorkerDaemon(config=config)
        joined = daemon.join("127.0.0.1")
        assert joined is True

        # Controller should have registered the worker
        time.sleep(0.5)
        workers = controller._heartbeat_listener.monitor.get_all_workers()
        assert len(workers) == 1
        assert workers[0].node.node_id == daemon.node.node_id

        controller._heartbeat_listener.stop()

    def test_worker_heartbeat(self):
        """Worker sends heartbeat and controller tracks it."""
        listener = HeartbeatListener(port=18901)
        listener.start()

        import socket
        msg = json.dumps({
            "type": "heartbeat",
            "node_id": "test-worker",
            "status": "idle",
            "compute_score": 100.0,
            "address": "127.0.0.1",
        }).encode()
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.settimeout(2.0)
            s.sendto(msg, ("127.0.0.1", 18901))

        time.sleep(0.5)
        assert listener.monitor.worker_count == 0  # heartbeat without join doesn't register
        listener.stop()


# --- Sprint 3: Task Queue ---

class TestTaskQueue:
    def test_submit_and_get(self):
        q = TaskQueue()
        t = Task(task_id="t1", task_type="test")
        assert q.submit(t) is True
        assert q.get_next().task_id == "t1"

    def test_priority_ordering(self):
        q = TaskQueue()
        q.submit(Task(task_id="low", task_type="test", priority=Priority.LOW))
        q.submit(Task(task_id="critical", task_type="test", priority=Priority.CRITICAL))
        q.submit(Task(task_id="normal", task_type="test", priority=Priority.NORMAL))
        assert q.get_next().task_id == "critical"
        assert q.get_next().task_id == "normal"
        assert q.get_next().task_id == "low"

    def test_duplicate_rejected(self):
        q = TaskQueue()
        q.submit(Task(task_id="t1", task_type="test"))
        assert q.submit(Task(task_id="t1", task_type="test")) is False

    def test_complete_task(self):
        q = TaskQueue()
        q.submit(Task(task_id="t1", task_type="test"))
        q.get_next()
        assert q.complete("t1", result="done") is True
        assert q.pending_count == 0


# --- Sprint 3: GPU Allocator ---

class TestGPUAllocator:
    def test_single_gpu_fit(self):
        node = make_node("n1", vram_mb=12288)
        allocator = GPUAllocator([node])
        model = ModelRequirements(model_name="phi-2", vram_mb=512, ram_mb=2048, quantization_levels=["int4"])
        plan = allocator.allocate(model)
        assert plan is not None
        assert plan.fits_on_single_gpu is True
        assert plan.node_count == 1

    def test_split_needed(self):
        node1 = make_node("n1", vram_mb=8192)
        node2 = make_node("n2", vram_mb=8192)
        allocator = GPUAllocator([node1, node2])
        model = ModelRequirements(model_name="llama-7b", vram_mb=8000, ram_mb=16000, quantization_levels=["fp16"])
        plan = allocator.allocate(model)
        assert plan is not None
        assert plan.fits_on_single_gpu is False
        assert plan.requires_split is True
        assert plan.node_count == 2

    def test_cpu_fallback(self):
        node = make_node("n1", vram_mb=0, ram_mb=32000)
        allocator = GPUAllocator([node])
        model = ModelRequirements(
            model_name="llama-7b", vram_mb=16000, ram_mb=16000,
            quantization_levels=["fp16"], supports_cpu_fallback=True,
        )
        plan = allocator.allocate(model)
        assert plan is not None
        assert plan.uses_cpu_fallback is True

    def test_cluster_capacity(self):
        node1 = make_node("n1", cpu_cores=8, ram_mb=16384, vram_mb=12288)
        node2 = make_node("n2", cpu_cores=4, ram_mb=8192, vram_mb=6144)
        allocator = GPUAllocator([node1, node2])
        cap = allocator.get_cluster_capacity()
        assert cap["total_vram_mb"] == 18432
        assert cap["total_cpu_cores"] == 12
        assert cap["total_ram_mb"] == 24576
        assert cap["node_count"] == 2

    def test_estimate_vram(self):
        vram_fp16 = estimate_model_vram("llama-7b", "fp16")
        vram_int4 = estimate_model_vram("llama-7b", "int4")
        assert vram_fp16 > vram_int4
        assert vram_int4 > 0


# --- Sprint 3: Scheduler ---

class TestScheduler:
    def test_register_and_schedule(self):
        node1 = make_node("n1", cpu_cores=16, ram_mb=32768, vram_mb=12288)
        node2 = make_node("n2", cpu_cores=8, ram_mb=16384, vram_mb=6144)
        scheduler = TaskScheduler()
        scheduler.register_node(node1)
        scheduler.register_node(node2)

        # Small task (4GB VRAM) — scheduler prefers tight-fit node (n2 with 6GB)
        task = Task(
            task_id="t1", task_type="llm_inference",
            priority=Priority.HIGH, required_ram_mb=4096,
            required_vram_mb=4000, required_cpu_cores=4,
        )
        scheduler.submit_task(task)
        assignment = scheduler.schedule_next()
        assert assignment is not None
        assert assignment.status == "assigned"
        assert assignment.node_id == "n2"  # Tight fit prefers 6GB over 12GB

        # Large task (8GB VRAM) — only n1 has enough
        task2 = Task(
            task_id="t2", task_type="llm_inference",
            priority=Priority.HIGH, required_ram_mb=16384,
            required_vram_mb=8000, required_cpu_cores=4,
        )
        scheduler.submit_task(task2)
        assignment2 = scheduler.schedule_next()
        assert assignment2.status == "assigned"
        assert assignment2.node_id == "n1"  # Only n1 has 8GB+

    def test_cpu_only_task(self):
        node = make_node("n1", vram_mb=0, ram_mb=8192, cpu_cores=8)
        scheduler = TaskScheduler()
        scheduler.register_node(node)
        task = Task(task_id="t1", task_type="batch_compute", required_ram_mb=2048, required_cpu_cores=4)
        scheduler.submit_task(task)
        assignment = scheduler.schedule_next()
        assert assignment.status == "assigned"

    def test_insufficient_resources(self):
        node = make_node("n1", vram_mb=0, ram_mb=1024, cpu_cores=2)
        scheduler = TaskScheduler()
        scheduler.register_node(node)
        task = Task(task_id="t1", task_type="test", required_ram_mb=8192, required_vram_mb=0)
        scheduler.submit_task(task)
        assignment = scheduler.schedule_next()
        assert assignment.status == "queued"


# --- Sprint 3: Task Types ---

class TestTaskTypes:
    def test_llm_inference_to_task(self):
        req = LLMInferenceRequest(model_name="phi-2", prompt="Hello!", max_tokens=100)
        task = req.to_task("task-001")
        assert task.task_type == "llm_inference"
        assert task.required_vram_mb > 0
        assert task.payload["model"] == "phi-2"
        assert task.payload["prompt"] == "Hello!"

    def test_batch_compute_to_task(self):
        req = BatchComputeRequest(job_name="render", command="blender", args=["-b", "scene.blend"])
        task = req.to_task("task-002")
        assert task.task_type == "batch_compute"
        assert task.payload["command"] == "blender"


# --- Sprint 4: CLI ---

class TestCLI:
    def test_help(self):
        result = subprocess.run(
            [sys.executable, "aggregatepc.py", "--help"],
            capture_output=True, text=True, timeout=5,
            cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        )
        assert result.returncode == 0
        assert "controller" in result.stdout
        assert "worker" in result.stdout

    def test_subcommand_help(self):
        base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        for cmd in ["controller", "worker", "profile", "status"]:
            result = subprocess.run(
                [sys.executable, "aggregatepc.py", cmd, "--help"],
                capture_output=True, text=True, timeout=5,
                cwd=base,
            )
            assert result.returncode == 0, f"{cmd} --help failed"


class TestConfig:
    def test_load_default_config(self):
        from cluster.config import load_config
        config = load_config()
        assert "controller_ip" in config
        assert "worker_ips" in config
        assert "controller_port" in config

    def test_load_missing_config_returns_defaults(self):
        from cluster.config import load_config
        config = load_config("/nonexistent/path.conf")
        assert config["controller_ip"] == "127.0.0.1"
        assert config["worker_ips"] == []

    def test_config_has_workers(self):
        from cluster.config import load_config
        config = load_config()
        assert len(config["worker_ips"]) >= 1


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-v"]))
