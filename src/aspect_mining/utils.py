import re
from .lexicon import STOPWORDS_LIGHT, NEGATIONS, INTENSIFIERS, SENTIMENT_LEXICON


def split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p.strip() for p in parts if p.strip()]


def simple_tokens(sentence: str) -> list[str]:
    return re.findall(r"[A-Za-z']+", sentence.lower())


def candidate_aspects_from_tokens(tokens: list[str]) -> list[str]:
    aspects = []
    for i, tok in enumerate(tokens):
        if tok in STOPWORDS_LIGHT or tok in SENTIMENT_LEXICON or tok in NEGATIONS or tok in INTENSIFIERS:
            continue
        if len(tok) < 3:
            continue
        # bi-gram compound noun heuristic
        if i + 1 < len(tokens):
            nxt = tokens[i + 1]
            if nxt not in STOPWORDS_LIGHT and nxt not in SENTIMENT_LEXICON and len(nxt) > 2:
                aspects.append(f"{tok} {nxt}")
        aspects.append(tok)
    return list(dict.fromkeys(aspects))
