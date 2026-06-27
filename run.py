#!/usr/bin/env python3
"""
Facial Expression Transfer — Cross-platform runner.
"""
import os
import sys


def main():
    project_root = os.path.dirname(os.path.abspath(__file__))

    # Linux: preload GLESv2 library
    if sys.platform == "linux":
        libs_dir = os.path.join(project_root, "libs")
        ld_path = os.environ.get("LD_LIBRARY_PATH", "")
        if libs_dir not in ld_path:
            new_ld_path = f"{libs_dir}:{ld_path}" if ld_path else libs_dir
            os.environ["LD_LIBRARY_PATH"] = new_ld_path

    sys.path.insert(0, project_root)
    main_script = os.path.join(project_root, "main.py")

    # Use subprocess for cross-platform compatibility
    import subprocess
    env = os.environ.copy()
    if sys.platform == "linux":
        libs_dir = os.path.join(project_root, "libs")
        if env.get("LD_LIBRARY_PATH"):
            env["LD_LIBRARY_PATH"] = f"{libs_dir}:{env['LD_LIBRARY_PATH']}"
        else:
            env["LD_LIBRARY_PATH"] = libs_dir

    return subprocess.call([sys.executable, main_script] + sys.argv[1:], env=env)


if __name__ == "__main__":
    sys.exit(main())
