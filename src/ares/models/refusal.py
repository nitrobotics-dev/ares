import re

REFUSAL_PHRASES = ["I'm sorry", "I'm unable"]


def check_refusal(text: str, refusal_phrases: list[str] | None = None) -> bool:
    if refusal_phrases is None:
        refusal_phrases = REFUSAL_PHRASES
    pattern = "|".join(map(re.escape, refusal_phrases))
    return bool(re.search(pattern, text))
