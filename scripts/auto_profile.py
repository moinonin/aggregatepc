"""One-click hardware and network profiling script.

Run this on each machine to generate a complete profile for cluster setup.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Ensure the project root is on the path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cluster.detect import detect_hardware
from cluster.nodes import create_local_node, NodeRole
from cluster.network.discovery import discover_peers


def main() -> None:
    parser = argparse.ArgumentParser(
        description="AggregatePC Auto-Profile: Detect hardware and discover cluster peers"
    )
    parser.add_argument(
        "--role",
        choices=["controller", "worker"],
        default="worker",
        help="Node role (default: worker)",
    )
    parser.add_argument(
        "--scan-timeout",
        type=float,
        default=3.0,
        help="Network scan timeout in seconds (default: 3.0)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file path (default: stdout)",
    )
    parser.add_argument(
        "--scan",
        action="store_true",
        help="Scan network for peers",
    )

    args = parser.parse_args()

    print(f"[aggregatepc] Profiling local hardware...")
    node = create_local_node(role=NodeRole(args.role))
    profile = node.to_dict()

    if args.scan:
        print(f"[aggregatepc] Scanning network for peers (timeout={args.scan_timeout}s)...")
        peers = discover_peers(timeout=args.scan_timeout)
        profile["peers"] = [
            {"service_name": p.service_name, "address": p.address, "port": p.port}
            for p in peers
        ]
        print(f"[aggregatepc] Found {len(peers)} peer(s)")

    output = json.dumps(profile, indent=2)

    if args.output:
        Path(args.output).write_text(output)
        print(f"[aggregatepc] Profile saved to {args.output}")
    else:
        print(output)


if __name__ == "__main__":
    main()
