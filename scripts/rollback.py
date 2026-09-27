"""One-command rollback: restore the previous champion model.

uv run python scripts/rollback.py
"""

import subprocess
import sys

if __name__ == "__main__":
    sys.exit(subprocess.call([sys.executable, "scripts/promote.py", "rollback"]))
