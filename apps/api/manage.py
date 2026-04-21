#!/usr/bin/env python3
"""Django's command-line utility for administrative tasks."""

from __future__ import annotations

import os
from pathlib import Path
import sys


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo_root / "src"))
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "snapcal_api.settings")
    from django.core.management import execute_from_command_line

    execute_from_command_line(sys.argv)


if __name__ == "__main__":
    main()
