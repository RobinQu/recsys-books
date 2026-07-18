#!/usr/bin/env python3
"""Download or verify papers, datasets and viewer assets from the repo root."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from recsys_lab.resources import ensure_resources


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify", action="store_true", help="verify only; never access the network")
    parser.add_argument("--offline", action="store_true", help="use local Hugging Face cache only")
    parser.add_argument("--strict", action="store_true", help="fail when a required or explicitly selected resource is unavailable")
    parser.add_argument("--include-optional", action="store_true", help="also fetch full/gated/large optional resources")
    parser.add_argument("--kind", action="append", choices=["papers", "datasets", "vendor"])
    parser.add_argument("--id", action="append", help="initialize only a specific manifest ID")
    args = parser.parse_args()
    result = ensure_resources(
        download=not args.verify,
        strict=args.strict,
        offline=args.offline,
        kinds=args.kind,
        ids=args.id,
        include_optional=args.include_optional,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
