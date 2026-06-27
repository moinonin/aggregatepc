"""Unified CLI entry point for AggregatePC.

Usage:
    aggregatepc controller          # Start this machine as the cluster controller
    aggregatepc worker             # Start this machine as a worker
    aggregatepc profile            # Detect hardware and scan for cluster
    aggregatepc status             # Show cluster status (from controller)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))


def cmd_controller(args: argparse.Namespace) -> None:
    """Start the cluster controller."""
    from cluster.nodes.controller import ClusterController, get_local_ip
    from cluster.config import load_config

    file_config = load_config(args.config)
    port = args.port if args.port != 8765 else file_config.get("controller_port", 8765)

    print(f"[aggregatepc] Starting controller on port {port}...")
    print(f"[aggregatepc] Controller IP: {get_local_ip()}")

    workers = file_config.get("worker_ips", [])
    if workers:
        print(f"[aggregatepc] Config has {len(workers)} worker(s): {', '.join(workers)}")
        print(f"[aggregatepc] Workers can join with: aggregatepc worker")
    else:
        print(f"[aggregatepc] Workers can join with: aggregatepc worker --controller {get_local_ip()}")

    controller = ClusterController(port=port)
    controller.run_forever()


def cmd_worker(args: argparse.Namespace) -> None:
    """Start a worker node."""
    from cluster.nodes.worker import WorkerDaemon, WorkerConfig, IdleThreshold
    from cluster.network.discovery import discover_peers
    from cluster.config import load_config

    file_config = load_config(args.config)

    # Controller IP: CLI arg > config file > auto-detect
    controller = args.controller
    if controller is None:
        config_controller = file_config.get("controller_ip", "")
        if config_controller and config_controller != "127.0.0.1":
            controller = config_controller
            print(f"[aggregatepc] Using controller from config: {controller}")
        else:
            print("[aggregatepc] Scanning for controller...")
            peers = discover_peers(timeout=args.scan_timeout)
            if peers:
                controller = peers[0].address
                print(f"[aggregatepc] Found controller at {controller}")
            else:
                print("[aggregatepc] No controller found. Use --controller <IP> or set it in configs/cluster.conf")
                sys.exit(1)

    worker_config = WorkerConfig(
        controller_port=args.port,
        worker=IdleThreshold(
            cpu_percent_max=args.cpu_threshold,
            memory_percent_max=args.mem_threshold,
            idle_duration_seconds=args.idle_duration,
        ),
    )

    daemon = WorkerDaemon(config=worker_config)

    print(f"[aggregatepc] Joining controller at {controller}:{args.port}...")
    if daemon.join(controller):
        print("[aggregatepc] Joined! Contributing idle compute to the cluster.")
        print("[aggregatepc] Press Ctrl+C to stop.")
    else:
        print("[aggregatepc] Failed to join controller.")
        sys.exit(1)

    daemon.run_forever()


def cmd_profile(args: argparse.Namespace) -> None:
    """Profile hardware and optionally scan network."""
    from scripts.auto_profile import main as profile_main
    import sys as _sys

    # Build argv for auto_profile
    argv = ["auto_profile.py"]
    if args.scan:
        argv.append("--scan")
    if args.output:
        argv.extend(["--output", args.output])
    if args.scan_timeout != 3.0:
        argv.extend(["--scan-timeout", str(args.scan_timeout)])

    _sys.argv = argv
    profile_main()


def _get_local_ip() -> str:
    """Get the local IP address."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("8.8.8.8", 80))
            return s.getsockname()[0]
    except OSError:
        return "127.0.0.1"


def cmd_status(args: argparse.Namespace) -> None:
    """Query cluster status from controller."""
    import json
    import socket
    from cluster.config import load_config

    file_config = load_config(args.config)
    controller_addr = args.controller or file_config.get("controller_ip", "127.0.0.1")
    # Status queries go to the controller's main port (8765), not a separate port
    port = args.port if args.port != 8765 else file_config.get("controller_port", 8765)
    local_ip = _get_local_ip()

    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.settimeout(5.0)
            s.bind(("", 0))
            callback_port = s.getsockname()[1]
            msg = json.dumps({
                "type": "status_query",
                "status_callback": {"address": local_ip, "port": callback_port},
            }).encode()
            s.sendto(msg, (controller_addr, port))
            data, _ = s.recvfrom(8192)
            status = json.loads(data.decode())

            # Enhance output with cluster metrics
            workers = status.get("workers", [])
            if workers:
                total_cores = sum(w.get("hardware", {}).get("cpu_cores", 0) for w in workers)
                total_ram = sum(w.get("hardware", {}).get("ram_mb", 0) for w in workers)
                total_vram = sum(
                    sum(g.get("vram_mb", 0) for g in w.get("hardware", {}).get("gpus", []))
                    for w in workers
                )
                all_models = set()
                for w in workers:
                    for m in w.get("models", []):
                        all_models.add(m)

                status["cluster_metrics"] = {
                    "total_cpu_cores": total_cores,
                    "total_ram_mb": total_ram,
                    "total_ram_gb": round(total_ram / 1024, 1),
                    "total_vram_mb": total_vram,
                    "total_vram_gb": round(total_vram / 1024, 1),
                    "total_models": len(all_models),
                    "available_models": sorted(all_models),
                }

            print(json.dumps(status, indent=2))
    except socket.timeout:
        print(f"[aggregatepc] No response from controller at {controller_addr}:{port}")
        sys.exit(1)
    except Exception as e:
        print(f"[aggregatepc] Error: {e}")
        sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="aggregatepc",
        description="AggregatePC - Distributed heterogeneous compute for idle PCs",
        epilog="Examples:\n"
               "  aggregatepc controller              # Be the cluster controller\n"
               "  aggregatepc worker --controller 1.2.3.4  # Join a cluster\n"
               "  aggregatepc profile --scan          # Detect hardware + find cluster\n",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Global option: config file path
    parser.add_argument("--config", type=str, default=None, help="Config file (default: configs/cluster.conf)")

    # Controller subcommand
    p_controller = subparsers.add_parser("controller", help="Start as cluster controller")
    p_controller.add_argument("--port", type=int, default=8765, help="UDP port (default: from config or 8765)")

    # Worker subcommand
    p_worker = subparsers.add_parser("worker", help="Start as worker node")
    p_worker.add_argument("--controller", type=str, default=None, help="Controller IP (from config if omitted)")
    p_worker.add_argument("--port", type=int, default=8765, help="Controller port (default: from config or 8765)")
    p_worker.add_argument("--cpu-threshold", type=float, default=25.0, help="Max CPU %% for idle (default: 25.0)")
    p_worker.add_argument("--mem-threshold", type=float, default=75.0, help="Max memory %% for idle (default: 75.0)")
    p_worker.add_argument("--idle-duration", type=float, default=30.0, help="Seconds idle before work (default: 30.0)")
    p_worker.add_argument("--scan-timeout", type=float, default=3.0, help="Controller scan timeout (default: 3.0)")
    p_worker.add_argument("--no-idle-check", action="store_true", help="Accept work even when busy")

    # Profile subcommand
    p_profile = subparsers.add_parser("profile", help="Profile hardware and scan network")
    p_profile.add_argument("--scan", action="store_true", help="Also scan for cluster")
    p_profile.add_argument("--output", type=str, default=None, help="Save profile to file")
    p_profile.add_argument("--scan-timeout", type=float, default=3.0, help="Scan timeout (default: 3.0)")

    # Status subcommand
    p_status = subparsers.add_parser("status", help="Query cluster status")
    p_status.add_argument("--controller", type=str, default=None, help="Controller IP (from config if omitted)")
    p_status.add_argument("--port", type=int, default=8765, help="Controller port (default: from config or 8765)")

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(0)

    commands = {
        "controller": cmd_controller,
        "worker": cmd_worker,
        "profile": cmd_profile,
        "status": cmd_status,
    }
    commands[args.command](args)


if __name__ == "__main__":
    main()
