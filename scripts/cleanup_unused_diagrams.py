#!/usr/bin/env python
"""Report diagram files that no documentation page references.

`make identify_unused_diagrams` (and therefore `make clean_old_diagrams` and
`make clean`) has always invoked this script, but the file was missing from the
repository, so those targets failed outright.

The scan is deliberately read-only: it writes a report and leaves every file in
place, because "unreferenced" is a hint, not proof -- an image may be pulled in
by a template, a generated page or an external site.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".gif", ".svg", ".webp"}
TEXT_SUFFIXES = {".md", ".yml", ".yaml", ".html", ".js", ".css", ".py", ".sh"}
SKIP_DIRS = {".git", ".venv", "venv", "site", "node_modules", "__pycache__", "dist"}


def iter_files(root: Path, suffixes: set[str]):
    """Yield files under root with one of the given suffixes.

    Args:
        root: Directory to walk.
        suffixes: Lower-case file extensions to keep.

    Yields:
        Matching paths, skipping build and vendor directories.
    """
    for path in root.rglob("*"):
        if any(part in SKIP_DIRS for part in path.parts):
            continue
        if path.is_file() and path.suffix.lower() in suffixes:
            yield path


def find_referenced_names(root: Path) -> set[str]:
    """Collect every image file name mentioned anywhere in the project.

    Matching is by base name rather than by resolved path: pages reference the
    same image through relative paths, site-root paths and mkdocs redirects, and
    a name match keeps the report conservative (it never claims a referenced
    image is unused).

    Args:
        root: Project root to scan.

    Returns:
        The set of image file names that appear in any text file.
    """
    referenced: set[str] = set()
    name_pattern = re.compile(
        r"[\w\-.]+\.(?:png|jpe?g|gif|svg|webp)",
        re.IGNORECASE,
    )
    for path in iter_files(root, TEXT_SUFFIXES):
        try:
            content = path.read_text(errors="ignore")
        except OSError:
            continue
        referenced.update(match.lower() for match in name_pattern.findall(content))
    return referenced


def main() -> int:
    """Scan the project and write the unused-diagram report.

    Returns:
        Process exit code; always 0 so the report never breaks `make clean`.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parent.parent,
        help="Project root to scan (default: the repository root).",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("unused_diagrams_report.txt"),
        help="Where to write the report.",
    )
    args = parser.parse_args()

    root: Path = args.root
    docs_dir = root / "docs"
    if not docs_dir.is_dir():
        sys.stderr.write(f"No docs directory found at {docs_dir}\n")
        return 0

    referenced = find_referenced_names(root)
    images = sorted(iter_files(docs_dir, IMAGE_SUFFIXES))
    unused = [img for img in images if img.name.lower() not in referenced]

    total_bytes = sum(img.stat().st_size for img in unused)
    lines = [
        "Unused diagram report",
        "=====================",
        "",
        f"Scanned {len(images)} image(s) under {docs_dir.relative_to(root)}.",
        f"{len(unused)} appear unreferenced ({total_bytes / 1024:.1f} KiB).",
        "",
        "Nothing has been deleted. Review the list before removing anything:",
        "an image can still be used by a template or an external page.",
        "",
    ]
    lines.extend(str(img.relative_to(root)) for img in unused)
    args.report.write_text("\n".join(lines) + "\n")

    sys.stdout.write(
        f"Scanned {len(images)} image(s); {len(unused)} unreferenced. "
        f"Report written to {args.report}\n",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
