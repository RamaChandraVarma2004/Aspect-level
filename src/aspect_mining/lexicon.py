"""Small, explainable lexicon for internship-level rule systems."""

SENTIMENT_LEXICON = {
    "amazing": 2.5,
    "awesome": 2.2,
    "good": 1.5,
    "great": 2.0,
    "excellent": 2.4,
    "love": 2.3,
    "smooth": 1.2,
    "fast": 1.2,
    "reliable": 1.6,
    "decent": 0.8,
    "bad": -1.8,
    "poor": -1.9,
    "awful": -2.5,
    "terrible": -2.6,
    "slow": -1.4,
    "weak": -1.3,
    "laggy": -2.1,
    "disappointing": -2.2,
    "noisy": -1.2,
    "expensive": -1.1,
}

NEGATIONS = {"not", "never", "no", "hardly", "without", "isn't", "don't", "can't"}
INTENSIFIERS = {"very": 1.5, "really": 1.4, "extremely": 1.9, "quite": 1.2, "slightly": 0.8}

GENERIC_ASPECTS = {"thing", "things", "item", "product", "stuff", "experience", "issue", "problem"}
