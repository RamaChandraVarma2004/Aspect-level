from src.aspect_miner import AspectOpinionPipeline


def test_mixed_aspects_have_expected_polarity():
    pipeline = AspectOpinionPipeline()
    text = "Battery life is great but screen quality is not good."
    result = pipeline.analyze(text)

    by_aspect = {item["aspect"]: item["sentiment"] for item in result["aspects"]}

    assert any("battery" in aspect and sentiment == "positive" for aspect, sentiment in by_aspect.items())
    assert any("screen" in aspect and sentiment == "negative" for aspect, sentiment in by_aspect.items())


def test_neutral_when_no_opinion_words_close():
    pipeline = AspectOpinionPipeline()
    text = "The phone has a camera and speaker."
    result = pipeline.analyze(text)
    assert all(item["sentiment"] == "neutral" for item in result["aspects"])
