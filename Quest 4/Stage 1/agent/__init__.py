"""Package initializer: put the starter kit on sys.path before any submodule runs.

Python guarantees this executes before agent.config / agent.loop / agent.run,
so `import mock_services` works in every submodule regardless of import order —
an auto-formatter can no longer break the bootstrap by re-sorting imports.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "starter-kit"))