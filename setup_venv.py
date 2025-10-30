#!/usr/bin/env python3
import os
import sys
import subprocess
from pathlib import Path

def run(cmd, check=True):
    """Run a shell command and print it first."""
    print(f"$ {' '.join(cmd)}")
    return subprocess.run(cmd, check=check)

def find_python(version: str) -> str:
    """Try to locate a Python interpreter for the given version."""
    candidates = [f"python{version}", f"python{version[0]}"]
    for c in candidates:
        path = subprocess.run(["which", c], capture_output=True, text=True)
        if path.returncode == 0:
            return path.stdout.strip()
    return None

def main():
    if len(sys.argv) != 2:
        print("Usage: python setup_env.py <python_version>")
        sys.exit(1)

    version = sys.argv[1]  # e.g. "3.11"
    venv_path = Path(".venv")

    # 1. Find or install the requested Python
    python_path = find_python(version)
    if not python_path:
        print(f"❌ Python {version} not found on system.")
        print("👉 Please install it manually, e.g.:")
        print(f"   brew install python@{version}   # macOS with Homebrew")
        print(f"   apt-get install python{version} # Ubuntu/Debian")
        sys.exit(1)

    print(f"✅ Found Python {version} at {python_path}")

    # 2. Create venv if not exists
    if not venv_path.exists():
        run([python_path, "-m", "venv", ".venv"])
        print("✅ Virtual environment created at .venv")
    else:
        print("ℹ️  .venv already exists, skipping creation")

    # 3. Install requirements.txt if present
    pip_path = str(venv_path / "bin" / "pip")
    if not Path("requirements.txt").exists():
        print("⚠️ No requirements.txt found, skipping pip install")
    else:
        run([pip_path, "install", "-r", "requirements.txt"])
        print("✅ Requirements installed")


if __name__ == "__main__":
    main()