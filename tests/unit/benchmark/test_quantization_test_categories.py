"""Documented pytest marker filters must retain every quantization test."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit
ROOT = Path(__file__).resolve().parents[3]
INTEGRATION_PATHS = [
    str(path.relative_to(ROOT))
    for pattern in ("test_kv_quantization*.py", "test_kivi_reference_flow.py")
    for path in sorted((ROOT / "tests/integration").glob(pattern))
]


@pytest.mark.parametrize(
    "category,paths",
    [
        ("unit", ["tests/unit/kv_quantization"]),
        ("integration", INTEGRATION_PATHS),
    ],
)
def test_category_filter_keeps_the_same_real_test_cases(category, paths, tmp_path):
    output = tmp_path / "collection.json"
    script = """
import json,sys
import pytest
class Capture:
    def pytest_collection_finish(self,session):
        self.ids=[item.nodeid for item in session.items]
results=[]
for marker in [[],["-m",sys.argv[2]]]:
    capture=Capture()
    result=pytest.main(["--collect-only","-q","-p","no:cacheprovider",*marker,*sys.argv[3:]],plugins=[capture])
    if result: raise SystemExit(result)
    results.append(capture.ids)
with open(sys.argv[1],"w") as file: json.dump(results,file)
"""
    env = {
        **os.environ,
        "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
    }
    result = subprocess.run(
        [sys.executable, "-c", script, str(output), category, *paths],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=45,
    )
    assert result.returncode == 0, result.stdout[-4000:] + result.stderr[-2000:]
    unfiltered, filtered = json.loads(output.read_text())
    assert unfiltered
    assert set(unfiltered) == set(filtered)
