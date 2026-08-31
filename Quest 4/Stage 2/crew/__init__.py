"""Package initializer: put the starter kit on sys.path before any submodule runs.

Same structural fix as Part A: Python guarantees a package's __init__ executes
before any of its submodules, so `import mock_services` / `import multi_agent_tools`
work everywhere in `crew.*` no matter how a formatter orders the imports.
"""

import sys
from pathlib import Path

STAGE_ROOT = Path(__file__).resolve().parent.parent
KIT_DIR = STAGE_ROOT / "starter-kit"

if str(KIT_DIR) not in sys.path:
    sys.path.insert(0, str(KIT_DIR))
