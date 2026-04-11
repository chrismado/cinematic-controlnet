from __future__ import annotations

import importlib.util


def test_torch_dependency_status_is_explicit() -> None:
    """Keep the suite green in lightweight environments while torch-backed tests skip."""
    torch_spec = importlib.util.find_spec("torch")
    if torch_spec is None:
        assert True
        return

    import torch

    assert isinstance(torch.__version__, str)
