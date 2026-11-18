"""
Package test suite.

Verifies fundamental package metadata and version attributes.
"""

import nrnet


def test_package_version() -> None:
    """
    Test that nrnet defines the expected semantic version string.

    :return: None.
    :rtype: None
    """
    assert nrnet.__version__ == "0.1.0"
