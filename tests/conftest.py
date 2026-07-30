"""
Pytest configuration: adds src/ to sys.path so tests can `import detector`,
`import embedding`, etc. exactly like the modules import each other,
without needing to package the project as an installable module.
"""

import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
