#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--template_motion", required=True, type=Path)
    ap.add_argument("--output_motion", required=True, type=Path)
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()

    src = args.template_motion
    dst = args.output_motion

    if not src.exists() or not src.is_dir():
        raise SystemExit(f"template_motion does not exist or is not a directory: {src}")
    if src.resolve() == dst.resolve():
        raise SystemExit("output_motion cannot be the same as template_motion")
    if dst.exists():
        raise SystemExit(f"output_motion already exists, refusing to overwrite: {dst}")

    print(f"Copy template motion\n  from: {src}\n  to:   {dst}")
    if args.dry_run:
        print("[dry_run] no files copied")
        return

    shutil.copytree(src, dst)
    print("Done.")


if __name__ == "__main__":
    main()
