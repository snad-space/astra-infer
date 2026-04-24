import astra_infer


def test_version():
    """Check to see that we can get the package version"""
    assert astra_infer.__version__ is not None
