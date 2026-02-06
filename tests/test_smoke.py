from pathlib import Path


def test_readme_mentions_four_versions():
    text = Path('README.md').read_text()
    assert '4 Versions' in text
