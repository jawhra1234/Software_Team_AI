"""Skeleton smoke test: the application package is importable and versioned."""

import app


def test_package_importable_and_versioned() -> None:
    assert isinstance(app.__version__, str)
    assert app.__version__ == "0.1.0"


def test_core_and_providers_subpackages_importable() -> None:
    import importlib

    for name in ("app.core", "app.providers"):
        module = importlib.import_module(name)
        assert module is not None
