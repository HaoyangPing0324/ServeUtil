"""Verify the frozen viewer against a real model report, then preserve executable modes."""

import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

root = Path("dist").resolve()
report = Path("examples/h20-decode.html").resolve()
if sys.platform == "darwin":
    folder = root / "ServeUtil.app"
    program = folder / "Contents" / "MacOS" / "ServeUtil"
else:
    folder = root / "ServeUtil"
    program = folder / ("ServeUtil.exe" if os.name == "nt" else "ServeUtil")
subprocess.run([str(program), "--verify", str(report)], check=True, timeout=120)
subprocess.run(
    [str(program), "--verify", str(report.with_name("decode.synthetic.json"))],
    check=True,
    timeout=120,
)
name = f"ServeUtil-{platform.system()}-{platform.machine()}"
shutil.make_archive(
    str(root / name), "zip" if os.name == "nt" else "gztar", root, folder.name
)
