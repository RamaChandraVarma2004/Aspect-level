SENTIMENT_LEXICON = {
    "amazing": 2.5, "awesome": 2.3, "good": 1.5, "great": 2.0, "excellent": 2.4,
    "love": 2.4, "like": 1.4, "smooth": 1.2, "responsive": 1.3, "fast": 1.1,
    "clear": 1.0, "bright": 1.0, "reliable": 1.7, "solid": 1.2, "decent": 0.7,
    "bad": -1.8, "poor": -1.9, "awful": -2.5, "terrible": -2.6, "weak": -1.3,
    "slow": -1.4, "laggy": -2.1, "blurry": -1.4, "dull": -1.2, "noisy": -1.2,
    "expensive": -1.0, "cheap": 0.8, "disappointing": -2.0,
}

NEGATIONS = {"not", "never", "no", "hardly", "rarely", "without", "isn't", "wasn't", "don't", "didn't", "can't"}
INTENSIFIERS = {"very": 1.5, "extremely": 1.9, "really": 1.4, "too": 1.3, "quite": 1.2, "slightly": 0.7, "somewhat": 0.8, "barely": 0.6}

GENERIC_ASPECTS = {"thing", "things", "item", "product", "stuff", "time", "day", "issue", "experience"}
STOPWORDS_LIGHT = {
    "the", "a", "an", "is", "are", "was", "were", "to", "for", "of", "in", "on", "at", "and", "or", "but", "it", "this", "that"
}
