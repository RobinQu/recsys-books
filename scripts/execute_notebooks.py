from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app.content import NOTEBOOKS


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", default="smoke", choices=["smoke", "full"])
    parser.add_argument("--only", help="只执行一个 Notebook slug")
    args = parser.parse_args()
    os.environ["RECSYS_PROFILE"] = args.profile
    slugs = [item["slug"] for item in NOTEBOOKS]
    summary_slugs = {"3_1_summary", "3_2_summary", "3_3_summary", "3_4_summary", "3_5_summary", "4_1_generative_overview"}
    execution_order = [slug for slug in slugs if slug not in summary_slugs] + [slug for slug in slugs if slug in summary_slugs]
    paths = [ROOT / "notebooks" / f"{slug}.ipynb" for slug in execution_order]
    if args.only:
        paths = [ROOT / "notebooks" / f"{args.only}.ipynb"]
    for path in paths:
        nb = nbformat.read(path, as_version=4)
        client = NotebookClient(nb, timeout=600 if args.profile == "smoke" else 7200, kernel_name="python3", resources={"metadata": {"path": str(ROOT / "notebooks")}})
        client.execute()
        nbformat.write(nb, path)
        print(f"executed {path.name}")


if __name__ == "__main__":
    main()
