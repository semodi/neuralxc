"""
Unit and regression test for the neuralxc package.
"""

# Import package, test suite, and other packages as needed
import neuralxc
import pytest
import sys

def test_neuralxc_imported():
    """Sample test, will always pass so long as import statement worked"""
    assert "neuralxc" in sys.modules
