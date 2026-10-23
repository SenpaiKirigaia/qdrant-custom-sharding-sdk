import subprocess

import pytest

pytestmark = pytest.mark.typing


def test_pyright_hides_upsert_helper():
    completed = subprocess.run(
        ["uv", "run", "--with", "pyright", "pyright", "tests/typing/check_private_hidden.py"],
        capture_output=True,
        text=True,
    )
    text = completed.stdout + completed.stderr
    assert completed.returncode != 0
    assert text.count('"_upsert"') >= 2
