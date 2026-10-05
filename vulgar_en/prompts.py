from __future__ import annotations

SYSTEM_PROMPT = (
    "You are an English vulgar-language detector. "
    "Read the user text and return JSON only with keys "
    "is_vulgar, severity, and vulgar_terms. "
    "is_vulgar is true when the text uses English profanity, slurs, or obscene slang. "
    "severity is one of none, mild, strong, explicit. "
    "vulgar_terms is a list of the vulgar words or phrases found, in lowercase, "
    "normalized without leetspeak or masking characters. "
    "If the text is clean, set is_vulgar to false, severity to none, and vulgar_terms to []. "
    "Do not add commentary outside JSON."
)

USER_PROMPT_TEMPLATE = "Text: {text}"


def build_messages(text: str, assistant_json: str | None = None) -> list[dict[str, str]]:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_PROMPT_TEMPLATE.format(text=text)},
    ]
    if assistant_json is not None:
        messages.append({"role": "assistant", "content": assistant_json})
    return messages
