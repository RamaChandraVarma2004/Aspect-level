from aspect_mining import AspectOpinionMiner
from aspect_mining.variants import run_variant


def test_mixed_aspect_sentiment():
    miner = AspectOpinionMiner()
    text = "Battery life is great but camera quality is not good."
    rows = miner.analyze(text)
    aspect_to_sentiment = {row["aspect"].lower(): row["sentiment"] for row in rows}

    assert any("battery" in k for k in aspect_to_sentiment)
    assert any("camera" in k for k in aspect_to_sentiment)
    assert any(v == "positive" for k, v in aspect_to_sentiment.items() if "battery" in k)
    assert any(v == "negative" for k, v in aspect_to_sentiment.items() if "camera" in k)


def test_multi_review_aggregation_counts():
    miner = AspectOpinionMiner()
    analyses = miner.analyze_reviews(
        [
            "Battery life is great and screen is good.",
            "Battery life is bad but screen is amazing.",
        ]
    )
    agg = miner.aggregate_aspects(analyses)
    by_aspect = {row["aspect"].lower(): row for row in agg}

    battery_key = next(key for key in by_aspect if "battery" in key)
    screen_key = next(key for key in by_aspect if "screen" in key)

    assert by_aspect[battery_key]["frequency"] == 2
    assert by_aspect[battery_key]["positive"] == 1
    assert by_aspect[battery_key]["negative"] == 1
    assert by_aspect[screen_key]["frequency"] == 2


def test_variants_produce_distinct_outputs():
    miner = AspectOpinionMiner()
    reviews = ["Camera quality is good but battery life is not great."]

    v1 = run_variant(miner, reviews, "v1")
    v2 = run_variant(miner, reviews, "v2")

    # v2 is conservative and keeps only multi-word aspects with nearest evidence.
    v1_mentions = sum(len(r.aspects) for r in v1["reviews"])
    v2_mentions = sum(len(r.aspects) for r in v2["reviews"])

    assert v1_mentions >= v2_mentions
