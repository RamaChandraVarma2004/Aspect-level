from aspect_mining.pipeline import AspectOpinionMiner


def test_supported_versions_declared():
    assert set(AspectOpinionMiner.ASSOCIATORS.keys()) == {"v1", "v2", "v3", "v4"}


def test_invalid_version_raises():
    try:
        AspectOpinionMiner(version="v99")
    except ValueError as exc:
        assert "Unsupported version" in str(exc)
    else:
        raise AssertionError("Expected ValueError for invalid version")
