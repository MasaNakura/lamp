"""Ensure the upstream LaMP training package (LaMP/LaMP) is importable."""
from __future__ import annotations

import os
import sys


def ensure_lamp_on_path(repo_root: str | None = None) -> str:
    """
    Insert LaMP/LaMP (contains `prompts`, `data`, `metrics`) at the front of sys.path.

    That directory should come from a **git submodule** at ``LaMP/`` (see README), or an
    equivalent manual clone of https://github.com/LaMP-Benchmark/LaMP into ``LaMP/``.
    """
    if repo_root is None:
        # util/lamp_paths.py -> repository root (sibling of util/, contains LaMP/)
        here = os.path.dirname(os.path.abspath(__file__))
        repo_root = os.path.abspath(os.path.join(here, ".."))
    lamp_pkg = os.path.join(repo_root, "LaMP", "LaMP")
    if not os.path.isdir(lamp_pkg):
        raise FileNotFoundError(
            f"Missing LaMP Python package at:\n  {lamp_pkg}\n\n"
            "From the repository root, fetch the benchmark as a submodule (recommended):\n"
            "  git submodule update --init --recursive\n"
            "Or add it once, then commit:\n"
            "  git submodule add https://github.com/LaMP-Benchmark/LaMP.git LaMP\n\n"
            "Alternatively clone that repo into ./LaMP so that ./LaMP/LaMP/ exists."
        )
    if lamp_pkg not in sys.path:
        sys.path.insert(0, lamp_pkg)
    return lamp_pkg
