#!/usr/bin/env python3
"""Pre-fetch robot binary assets (meshes) from Hugging Face into their project paths.

Robot STL meshes are hosted on Hugging Face rather than committed to git. They are
also downloaded automatically on first use, but this command lets you pull them
ahead of time (e.g. for CI or offline prep) with a single invocation. Files land
under ``src/wanwanlab/assets/robots/<robot>/meshes/`` — no manual file moving needed.

Usage:
  uv run wanwanlab-pull-assets               # pull the default robot (x2)
  uv run wanwanlab-pull-assets --robot x2
"""

from __future__ import annotations

import argparse
import logging
from collections.abc import Sequence

from wanwanlab.assets.hub import resolve_robot_asset_dir
