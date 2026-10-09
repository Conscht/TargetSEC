"""Test1 must never reach a training loader.

An earlier revision pointed every decoder-stage loader at the test partition,
which invalidated every number produced for months. src/dataset.py now carries
an explicit guard; this asserts the guard is still wired up.
"""
import pytest

import src.dataset as ds


def test_train_and_val_are_not_the_test_partition():
    assert ds.DEFAULT_TENSOR_DIR != ds.DEFAULT_TEST_TENSOR_DIR
    assert ds.DEFAULT_VAL_TENSOR_DIR != ds.DEFAULT_TEST_TENSOR_DIR


def test_guard_rejects_the_test_partition():
    with pytest.raises(ValueError, match="(?i)test"):
        ds._assert_no_test_leak(ds.DEFAULT_TEST_TENSOR_DIR)


def test_guard_allows_train_and_development():
    ds._assert_no_test_leak(ds.DEFAULT_TENSOR_DIR, ds.DEFAULT_VAL_TENSOR_DIR)


def test_data_root_is_configurable():
    """Paths must come from TARGETSEC_DATA_ROOT so a clone is not pinned to
    the cluster this was developed on."""
    import importlib, os

    os.environ["TARGETSEC_DATA_ROOT"] = "/tmp/targetsec-test-root"
    try:
        importlib.reload(ds)
        assert ds.DEFAULT_TENSOR_DIR.startswith("/tmp/targetsec-test-root")
    finally:
        del os.environ["TARGETSEC_DATA_ROOT"]
        importlib.reload(ds)
