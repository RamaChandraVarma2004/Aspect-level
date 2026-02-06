"""Explainable resources shared across all approaches."""

SENTIMENT_LEXICON = {
    "amazing": 2.5,
    "awesome": 2.3,
    "good": 1.6,
    "great": 2.0,
    "fast": 1.2,
    "smooth": 1.3,
    "responsive": 1.4,
    "decent": 0.8,
    "excellent": 2.4,
    "nice": 1.2,
    "love": 2.3,
    "reliable": 1.7,
    "durable": 1.8,
    "clear": 1.2,
    "bad": -1.8,
    "poor": -1.9,
    "slow": -1.4,
    "laggy": -2.1,
    "awful": -2.5,
    "terrible": -2.6,
    "weak": -1.2,
    "expensive": -1.0,
    "disappointing": -2.2,
    "noisy": -1.2,
    "dim": -1.2,
    "heavy": -0.8,
    "blurry": -1.6,
}

NEGATIONS = {
    "not", "never", "no", "hardly", "rarely", "without", "isn't", "wasn't", "don't", "didn't", "can't"
}

INTENSIFIERS = {
    "very": 1.5,
    "extremely": 1.9,
    "too": 1.3,
    "really": 1.4,
    "quite": 1.2,
    "slightly": 0.7,
    "somewhat": 0.8,
    "barely": 0.6,
}

GENERIC_ASPECTS = {
    "thing", "things", "item", "product", "stuff", "time", "day", "issue", "problem", "experience"
}
