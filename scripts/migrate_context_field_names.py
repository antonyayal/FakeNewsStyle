# scripts/migrate_context_field_names.py
# -*- coding: utf-8 -*-
"""
One-off migration: renames the context-branch field names inside historical
results/*.json records to match the Source Name/Source Link/Domain naming
adopted across the codebase (see src/features/context_extractor.py).

Old -> new:
  context_dims.source -> context_dims.source_name
  context_dims.domain -> context_dims.source_link
  top-level key topic_breakdown -> domain_breakdown

Scope: results/*.json only (the active results folder). results_old/,
results_old_2/, results_old_3/ are frozen/superseded archives and are
intentionally left untouched.

Usage:
    python scripts/migrate_context_field_names.py            # dry run, prints a summary
    python scripts/migrate_context_field_names.py --apply     # writes the changes
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = BASE_DIR / "results"


def migrate_record(data: dict) -> tuple[dict, bool]:
    changed = False

    cd = data.get("context_dims")
    if isinstance(cd, dict):
        new_cd = dict(cd)
        if "source" in new_cd:
            new_cd["source_name"] = new_cd.pop("source")
            changed = True
        if "domain" in new_cd:
            new_cd["source_link"] = new_cd.pop("domain")
            changed = True
        data["context_dims"] = new_cd

    if "topic_breakdown" in data:
        data["domain_breakdown"] = data.pop("topic_breakdown")
        changed = True

    return data, changed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true", help="Write changes (default: dry run)")
    args = parser.parse_args()

    files = sorted(RESULTS_DIR.glob("*.json"))
    n_total = len(files)
    n_changed = 0
    n_unreadable = 0
    sample_diffs = []

    for f in files:
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
        except Exception as exc:
            n_unreadable += 1
            print(f"  SKIP (unreadable): {f.name}: {exc}")
            continue

        if not isinstance(data, dict):
            continue  # e.g. per_fold files shaped as {"configs": {...}} handled by inspection, not blind key-walk

        new_data, changed = migrate_record(data)
        if changed:
            n_changed += 1
            if len(sample_diffs) < 3:
                sample_diffs.append(f.name)
            if args.apply:
                f.write_text(json.dumps(new_data, indent=2, ensure_ascii=False), encoding="utf-8")

    mode = "APPLIED" if args.apply else "DRY RUN"
    print(f"\n[{mode}] {RESULTS_DIR}: {n_total} files scanned, {n_changed} would change"
          f"{'d' if args.apply else ''}, {n_unreadable} unreadable.")
    if sample_diffs:
        print(f"Sample affected files: {sample_diffs}")
    if not args.apply and n_changed:
        print("Re-run with --apply to write the changes.")


if __name__ == "__main__":
    main()
