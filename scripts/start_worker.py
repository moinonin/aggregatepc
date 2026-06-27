"""Simple script to turn any PC into a cluster worker.

Usage:
    python3 scripts/start_worker.py                    # Start worker (auto-discover controller)
    python3 scripts/start_worker.py --controller 1.2.3.4  # Join specific controller
    python3 scripts/start_worker.py --cpu-threshold 15   # Stricter idle detection
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cluster.nodes.worker import WorkerDaemon, WorkerConfig, IdleThreshold
from cluster.network.discovery import discover_peers


def find_controller(timeout: float = 3.0) -> str | None:
    """Try to auto-discover a controller on the local network."""
    peers = discover_peers(timeout=timeout)
    if peers:
        # First peer that responds is our controller
        return peers[0].address
    return None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Start an AggregatePC worker on this machine"
    )
    parser.add_argument(
        "--controller",
        type=str,
        default=None,
        help="Controller IP address (auto-discover if not specified)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8765,
        help="Controller port (default: 8765)",
    )
    parser.add_argument(
        "--cpu-threshold",
        type=float,
        default=25.0,
        help="Max CPU %% to consider machine idle (default: 25.0)",
    )
    parser.add_argument(
        "--mem-threshold",
        type=float,
        default=75.0,
        help="Max memory %% to consider machine idle (default: 75.0)",
    )
    parser.add_argument(
        "--idle-duration",
        type=float,
        default=30.0,
        help="Seconds machine must be idle before accepting work (default: 30.0)",
    )
    parser.add_argument(
        "--no-idle-check",
        action="store_true",
        help="Accept work even when machine is in use (not recommended)",
    )

    args = parser.parse_args()

    # Find controller
    controller = args.controller
    if controller is None:
        print("[aggregatepc] Scanning for controller on local network...")
        controller = find_controller()
        if controller:
            print(f"[aggregatepc] Found controller at {controller}")
        else:
            print("[aggregatepc] No controller found. Use --controller <IP> to specify one.")
            sys.exit(1)

    # Configure worker
    if args.no_idle_check:
        threshold = IdleThreshold(
            cpu_percent_max=100.0,
            memory_percent_max=100.0,
            idle_duration_seconds=0.0,
        )
    else:
        threshold = IdleThreshold(
            cpu_percent_max=args.cpu_threshold,
            memory_percent_max=args.mem_threshold,
            idle_duration_seconds=args.idle_duration,
        )

    config = WorkerConfig(
        worker=threshold,
        controller_port=args.port,
    )

    daemon = WorkerDaemon(config=config)

    print(f"[aggregatepc] Joining controller at {controller}:{args.port}...")
    if daemon.join(controller):
        print("[aggregatepc] Joined! Contributing idle compute to the cluster.")
        print("[aggregatepc] Press Ctrl+C to stop.")
    else:
        print("[aggregatepc] Failed to join controller.")
        sys.exit(1)

    daemon.run_forever()


if __name__ == "__main__":
    main()
