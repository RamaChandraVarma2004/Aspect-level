from aspect_mining import AspectOpinionMiner


def test_v3_mixed_aspect_sentiment():
    miner = AspectOpinionMiner(version="v3")
    text = "Battery life is great but camera quality is not good."
    rows = miner.analyze(text)
    assert rows
    assert any(r["sentiment"] == "positive" for r in rows)
    assert any(r["sentiment"] == "negative" for r in rows)


def test_v4_runs_without_spacy_dependency():
    miner = AspectOpinionMiner(version="v4")
    rows = miner.analyze("Screen is amazing but speakers are weak.")
    assert isinstance(rows, list)
    assert all("aspect" in r and "sentiment" in r for r in rows)
