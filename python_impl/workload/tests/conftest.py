"""Make `import python_impl...` work when pytest is started from any folder."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]  # the folder that contains python_impl/
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
