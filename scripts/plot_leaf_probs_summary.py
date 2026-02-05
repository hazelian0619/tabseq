#!/usr/bin/env python3
"""
Compatibility wrapper: script moved to scripts/analysis/plot_leaf_probs_summary.py.
"""
from __future__ import annotations

import os
import runpy

HERE = os.path.dirname(__file__)
TARGET = os.path.join(HERE, "analysis", "plot_leaf_probs_summary.py")
runpy.run_path(TARGET, run_name="__main__")
