#!/usr/bin/env python
"""Upload MTEB(por, v2) eval results from the local mteb cache to the HF dataset.

Run this on a **cheap CPU-only instance AFTER** the GPU eval run — never during
the GPU run (don't burn expensive GPU time on uploads). The GPU run writes
results to ``MTEB_CACHE`` on the persistent block/SFS volume; this script then
batch-uploads them to the public results dataset on the Hub.

Usage::

    export HF_TOKEN=...
    export MTEB_CACHE=/mnt/vol/mteb_cache        # the same volume the GPU run used
    python scripts/upload_results_to_hf.py                 # upload
    python scripts/upload_results_to_hf.py --dry-run       # list only
    python scripts/upload_results_to_hf.py --repo mteb-pt/mteb-pt-results

Uses ``upload_folder`` (a single batched commit) to avoid the per-file 429
rate-limits seen with many parallel uploads.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from huggingface_hub import HfApi


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Batch-upload the mteb results cache to the HF results dataset.",
    )
    parser.add_argument("--repo", default="mteb-pt/mteb-pt-results", help="Target HF dataset repo")
    parser.add_argument(
        "--cache",
        default=os.environ.get("MTEB_CACHE", str(Path.home() / ".cache" / "mteb")),
        help="mteb results cache dir (default: $MTEB_CACHE or ~/.cache/mteb)",
    )
    parser.add_argument(
        "--path-in-repo", default="results", help="Destination prefix inside the repo"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="List what would upload, upload nothing"
    )
    args = parser.parse_args()

    results_dir = Path(args.cache) / "results"
    if not results_dir.is_dir():
        results_dir = Path(args.cache)  # fall back: the cache root is the results dir

    json_files = sorted(results_dir.rglob("*.json"))
    print(f"cache:  {results_dir}")
    print(f"found:  {len(json_files)} result JSON files")
    if not json_files:
        raise SystemExit("No result JSONs found — check --cache / MTEB_CACHE.")

    if args.dry_run:
        for f in json_files[:20]:
            print("  ", f.relative_to(results_dir))
        if len(json_files) > 20:
            print(f"  ... ({len(json_files)} total)")
        print("dry run — nothing uploaded.")
        return

    api = HfApi()
    api.create_repo(args.repo, repo_type="dataset", exist_ok=True)
    api.upload_folder(
        folder_path=str(results_dir),
        repo_id=args.repo,
        repo_type="dataset",
        path_in_repo=args.path_in_repo,
        allow_patterns=["*.json"],
        commit_message="Upload MTEB(por, v2) eval results",
    )
    print(f"uploaded {len(json_files)} files -> {args.repo}/{args.path_in_repo}")


if __name__ == "__main__":
    main()
