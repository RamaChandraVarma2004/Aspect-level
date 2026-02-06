import pytest

from aspect_mining import AspectOpinionMiner


def _require_spacy():
    pytest.importorskip("spacy")


@pytest.mark.parametrize(
    "approach",
    [
        "v1_nearby_window",
        "v2_dependency_rules",
        "v3_clause_aware",
        "v4_ensemble_consensus",
    ],
)
def test_each_version_returns_output(approach):
    _require_spacy()
    miner = AspectOpinionMiner(approach=approach)
    text = "Battery life is great but camera quality is not good."
    rows = miner.analyze(text)
    assert isinstance(rows, list)


def test_ensemble_has_expected_aspects():
    _require_spacy()
    miner = AspectOpinionMiner(approach="v4_ensemble_consensus")
    rows = miner.analyze("Battery life is great but camera quality is not good.")
    joined = " ".join(x["aspect"].lower() for x in rows)
    assert "battery" in joined
    assert "camera" in joined


def test_project_sanity_marker():
    assert True
