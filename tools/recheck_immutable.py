#!/usr/bin/env python3
"""Recheck every entry of reports/dev/housekeeping/immutable_initial_sha256.json.

Read-only. Usage (from anywhere):
    python3 -I recheck_immutable.py [--root ~/microduck/nav] [--list PATH] [--json OUT]
Prints total/ok/mismatch/missing/invalid counts and every failure; exit 0 only if
all entries exist and match, exit 1 on any mismatch/missing/invalid, exit 2 on
an unreadable or malformed list.
"""
import argparse
import hashlib
import json
import os
import re
import sys
from pathlib import Path

HEX64 = re.compile(r"^[0-9a-f]{64}$")


def sha256(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(Path.home() / "microduck" / "nav"))
    ap.add_argument("--list", default=None,
                    help="default: <root>/reports/dev/housekeeping/immutable_initial_sha256.json")
    ap.add_argument("--json", default=None, help="optional path for a machine-readable result")
    args = ap.parse_args(argv)

    root = Path(args.root).expanduser().resolve()
    list_path = Path(args.list).expanduser() if args.list else \
        root / "reports" / "dev" / "housekeeping" / "immutable_initial_sha256.json"
    try:
        expected = json.loads(list_path.read_text())
    except (OSError, ValueError) as e:
        print(f"ERROR: cannot read immutable list {list_path}: {e}", file=sys.stderr)
        return 2
    if not isinstance(expected, dict) or not expected:
        print(f"ERROR: {list_path} is not a non-empty {{path: sha256}} object", file=sys.stderr)
        return 2

    ok, mismatched, missing, invalid = [], [], [], []
    for rel, want in sorted(expected.items()):
        if not isinstance(want, str) or not HEX64.match(want):
            invalid.append((rel, f"bad expected hash {want!r}"))
            continue
        p = Path(rel)
        if p.is_absolute() or ".." in p.parts:
            invalid.append((rel, "path escapes root"))
            continue
        full = root / p
        if not full.is_file():
            missing.append(rel)
            continue
        try:
            got = sha256(full)
        except OSError as e:
            invalid.append((rel, f"unreadable: {e}"))
            continue
        (ok if got == want else mismatched).append((rel, want, got) if got != want else rel)

    total = len(expected)
    print(f"immutable list: {list_path}")
    print(f"root:           {root}")
    print(f"total={total} ok={len(ok)} mismatched={len(mismatched)} "
          f"missing={len(missing)} invalid={len(invalid)}")
    for rel, want, got in mismatched:
        print(f"MISMATCH {rel}\n  expected {want}\n  actual   {got}")
    for rel in missing:
        print(f"MISSING  {rel}")
    for rel, why in invalid:
        print(f"INVALID  {rel}: {why}")

    passed = len(ok) == total and not (mismatched or missing or invalid)
    if args.json:
        out = {"list": str(list_path), "root": str(root), "total": total, "ok": len(ok),
               "mismatched": [{"path": r, "expected": w, "actual": g} for r, w, g in mismatched],
               "missing": missing, "invalid": [{"path": r, "reason": y} for r, y in invalid],
               "passed": passed}
        Path(args.json).write_text(json.dumps(out, indent=2) + "\n")
    print("PASS: all immutable artifacts unchanged" if passed else "FAIL: immutable artifacts changed or missing")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
