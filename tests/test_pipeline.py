from aspect_miner.pipeline import AspectOpinionPipeline


def test_mixed_aspect_sentiment():
    pipeline = AspectOpinionPipeline()
    text = "Battery life is great but screen quality is not good."
    result = pipeline.analyze(text)

    by_aspect = {row["aspect"]: row["sentiment"] for row in result["aspect_results"]}

    assert any("battery" in aspect for aspect in by_aspect)
    assert any("screen" in aspect for aspect in by_aspect)

    battery_aspect = next(a for a in by_aspect if "battery" in a)
    screen_aspect = next(a for a in by_aspect if "screen" in a)

    assert by_aspect[battery_aspect] == "positive"
    assert by_aspect[screen_aspect] == "negative"
