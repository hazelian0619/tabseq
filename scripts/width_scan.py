#!/usr/bin/env python3
"""
Compatibility wrapper: script moved to scripts/analysis/width_scan.py.
"""
from __future__ import annotations

import os
import runpy

HERE = os.path.dirname(__file__)
TARGET = os.path.join(HERE, "analysis", "width_scan.py")
runpy.run_path(TARGET, run_name="__main__")
