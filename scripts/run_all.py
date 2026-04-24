from __future__ import annotations

import subprocess
import sys


SCRIPTS = [
    "scripts/run_data_validation.py",
    "scripts/run_training.py",
    "scripts/run_evaluation.py",
    "scripts/run_stream_simulation.py",
]


def main() -> None:
    for script in SCRIPTS:
        print(f"Running {script}...")
        result = subprocess.run([sys.executable, script], check=False)
        if result.returncode != 0:
            raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
