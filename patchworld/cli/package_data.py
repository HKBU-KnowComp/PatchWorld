from __future__ import annotations

import argparse
import hashlib
import json
import tarfile
from datetime import datetime
from pathlib import Path
from typing import Dict, List


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _count_jsonl(path: Path) -> Dict[str, int]:
    trajectories = 0
    transitions = 0
    with path.open("r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            trajectories += 1
            try:
                obj = json.loads(line)
                transitions += len(obj.get("transitions", []))
            except Exception:
                pass
    return {"trajectories": trajectories, "transitions": transitions}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Package PatchWorld trajectory splits for release.")
    parser.add_argument(
        "--source_root",
        required=True,
        help="Root directory containing per-env split JSONL files.",
    )
    parser.add_argument(
        "--envs",
        default="alfworld,babyai,maze,sciworld,textcraft,webshop,wordle",
        help="Comma-separated env names to include.",
    )
    parser.add_argument(
        "--output_dir",
        default="artifacts/patchworld/data_release",
        help="Output directory for packaged split files.",
    )
    parser.add_argument(
        "--archive",
        action="store_true",
        help="Also create a .tar.gz archive under output_dir/..",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_root = Path(args.source_root)
    if not source_root.exists():
        raise SystemExit(f"source root not found: {source_root}")

    envs = [e.strip() for e in args.envs.split(",") if e.strip()]
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest: Dict[str, object] = {
        "timestamp": datetime.now().isoformat(),
        "source_root": source_root.name,
        "output_dir": str(out_dir),
        "envs": envs,
        "files": [],
    }
    manifest_files: List[Dict[str, object]] = manifest["files"]  # type: ignore[assignment]

    total_files = 0
    for env in envs:
        src_env = source_root / env
        if not src_env.exists():
            raise SystemExit(f"missing env directory: {src_env}")
        dst_env = out_dir / env
        dst_env.mkdir(parents=True, exist_ok=True)
        files = sorted(src_env.glob(f"{env}_traj_*.jsonl"))
        if not files:
            raise SystemExit(f"no split files found for env={env} in {src_env}")
        for src in files:
            dst = dst_env / src.name
            dst.write_bytes(src.read_bytes())
            counts = _count_jsonl(dst)
            info = {
                "env": env,
                "relative_path": str(dst.relative_to(out_dir)),
                "bytes": dst.stat().st_size,
                "sha256": _sha256(dst),
                "trajectories": counts["trajectories"],
                "transitions": counts["transitions"],
            }
            manifest_files.append(info)
            total_files += 1

    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"[patchworld-package-data] wrote {total_files} files -> {out_dir}")
    print(f"[patchworld-package-data] manifest -> {manifest_path}")

    if args.archive:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        archive_path = out_dir.parent / f"{out_dir.name}_{ts}.tar.gz"
        with tarfile.open(archive_path, "w:gz") as tar:
            tar.add(out_dir, arcname=out_dir.name)
        print(f"[patchworld-package-data] archive -> {archive_path}")


if __name__ == "__main__":
    main()
