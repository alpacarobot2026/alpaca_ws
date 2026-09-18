#!/usr/bin/env python3
"""ros2 run entry point for TalkNCE active-speaker detection.

TalkNCE isn't colcon-installed like the rest of this package: dlhammer and
torchvggish do absolute imports (`import model.xxx`, `from torchvggish import
vggish`) that only resolve with the vendored talknce/ tree itself on
sys.path, which setuptools packaging doesn't preserve. See
alpaca_interaction/talknce/PROVENANCE.md.

Instead, this wrapper locates that source tree at runtime (same sibling
src/+install/ workspace-root trick alpaca_navigation/prediction/prediction.py
uses for the HST backend's un-installed assets), chdir's into it (the real
script's --cfg default is a path relative to cwd), and execs it in place.
"""

import importlib.util
import os
import sys
from pathlib import Path

from rclpy.utilities import remove_ros_args


def _find_workspace_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "src").is_dir() and (parent / "install").is_dir():
            return parent
    return Path(__file__).resolve().parents[4]


def main():
    talknce_dir = _find_workspace_root() / "src" / "alpaca_interaction" / "talknce"
    script_path = talknce_dir / "scripts" / "ros2_infer_live.py"
    os.chdir(talknce_dir)

    # Strip -r/-p/--ros-args before the vendored script's own argparse sees
    # them; ros2_infer_live.py has no rclpy.init(args=...) hook for these.
    sys.argv = remove_ros_args(sys.argv)

    spec = importlib.util.spec_from_file_location("talknce_ros2_infer_live", script_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.main()


if __name__ == "__main__":
    main()
