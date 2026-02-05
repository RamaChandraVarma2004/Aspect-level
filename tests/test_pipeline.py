from aspect_mining import AspectOpinionMiner


def test_mixed_aspect_sentiment():
    miner = AspectOpinionMiner()
    text = "Battery life is great but camera quality is not good."
    rows = miner.analyze(text)
    aspect_to_sentiment = {row["aspect"].lower(): row["sentiment"] for row in rows}

    assert any("battery" in k for k in aspect_to_sentiment)
    assert any("camera" in k for k in aspect_to_sentiment)
    assert any(v == "positive" for k, v in aspect_to_sentiment.items() if "battery" in k)
    assert any(v == "negative" for k, v in aspect_to_sentiment.items() if "camera" in k)
