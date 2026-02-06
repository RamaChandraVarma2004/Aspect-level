import pytest

spacy = pytest.importorskip('spacy', reason='spacy not installed')
from aspect_mining import AspectOpinionMiner


def test_invalid_version_raises():
    with pytest.raises(ValueError):
        AspectOpinionMiner(version='v9')


@pytest.mark.parametrize('version', ['v1', 'v2', 'v3', 'v4'])
def test_versions_extract_multi_aspect(version):
    miner = AspectOpinionMiner(version=version)
    text = 'Battery life is great but camera quality is not good.'
    rows = miner.analyze(text)

    aspects = [row['aspect'].lower() for row in rows]
    assert any('battery' in a for a in aspects)
    assert any('camera' in a for a in aspects)
