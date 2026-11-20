"""Small, conservative routes for requests that do not need manual retrieval."""

from __future__ import annotations

import re


GREETING_V2 = re.compile(r"^(?:hi|hello|hey|hiya|good (?:morning|afternoon|evening))\b[\s,!.]*"
                         r"(?:there|everyone|all|manuora|again)?[\s,!.]*$")
THANKS_V2 = re.compile(r"^(?:thanks|thank you|thx|ty|cheers|great,? thanks|ok,? thanks|perfect,? thanks)\b[^?]{0,40}$")


# Only a message that asks what the latest version *is*; one that asks about something in the latest version
# ("what changed between 28 and the latest version") goes to the router, which reads "latest" as the newest
# version in the library (user decision 2026-11-20).
WHICH_LATEST = re.compile(r"^(?:what|which)(?:'s| is)?(?: the)? (?:latest|newest|most recent|current)"
                          r"(?: (?P<p1>\w+))? (?:version|release)(?: of(?: the)? (?P<p2>\w+))?$"
                          r"|^is there an? (?:newer|new|later) (?:version|release)(?: of(?: the)? (?P<p3>\w+))?$")


def early_response(question: str, assistant_name: str, product_ids: set[str], rules: str = "v1",
                   newest: dict[str, str] | None = None) -> str | None:
    text = re.sub(r"[?!.\s]+$", "", question.casefold().strip())
    if rules == "v2":
        # Rules v2 (experiment R): greetings and thanks with a few extra words ("hi there!",
        # "thanks, that helped a lot"), as long as they ask nothing and name no product.
        names_product = any(re.search(rf"\b{re.escape(p)}\b", text) for p in product_ids)
        if not names_product and GREETING_V2.match(text):
            return f"Hello! I'm {assistant_name}. What can I help you find in the product manuals?"
        if not names_product and "?" not in question and THANKS_V2.match(text):
            return "You're welcome!"
    if text in {"what is your name", "what's your name", "who are you",
                "tell me your name", "what should i call you"}:
        return f"I am {assistant_name}, a support assistant for product manuals."
    if text in {"hi", "hello", "hey", "good morning", "good afternoon", "good evening"}:
        return f"Hello! I'm {assistant_name}. What can I help you find in the product manuals?"
    if text in {"thanks", "thank you", "thank you very much"}:
        return "You're welcome!"
    if text in {"how old are you", "when were you born", "do you have an age"}:
        return f"I don't have an age like a person. I'm {assistant_name}, a support assistant for product manuals."
    if text in {"what can you do", "how can you help me"}:
        return "I can help answer questions about the product manuals available here and show the source pages."

    # A weather feature is a support question; actual conditions require live data.
    weather_terms = r"\b(weather|forecast|temperature|rain(?:ing)?)\b"
    support_terms = r"\b(app|widget|feature|manual|guide|configure|enable|install|setup|setting|display|show)\b"
    support_action = r"\b(configure|enable|install|setup|display|show|see|use|change|open)\b"
    live_weather = r"\b(now|today|tomorrow|currently|right now|this week)\b"
    names_product = any(re.search(rf"\b{re.escape(product)}\b", text) for product in product_ids)
    if re.search(weather_terms, text):
        product_topic = names_product or re.search(support_terms, text)
        if not product_topic or (re.search(live_weather, text) and not re.search(support_action, text)):
            return "I don't have live weather information. I can help with questions about the product manuals here."

    if (match := WHICH_LATEST.match(text)):
        from manuora.api.grounding import PRODUCT_NAMES
        product = next((p for p in (match.group("p1"), match.group("p2"), match.group("p3")) if p in product_ids), None)
        if product and newest and newest.get(product):
            return (f"The newest {PRODUCT_NAMES.get(product, product)} manual in this library is version "
                    f"{newest[product]}. I can't tell from the manuals whether a newer release exists.")
        return "I can't verify the latest release from these manuals. I can help with a specific documented version."
    return None


def supported_evidence(evidence: list[dict], minimum_score: float) -> list[dict]:
    """Keep passages above a provisional relevance floor, preserving citation order."""
    return [hit for hit in evidence if hit["rerank_score"] >= minimum_score]


def answer_lacks_support(answer: str) -> bool:
    """Catch the generator's abstention marker and common unsupported claims.

    The marker counts anywhere: the model sometimes explains the gap first and
    ends with INSUFFICIENT_EVIDENCE, and that is still a refusal, not an answer.
    """
    text = answer.casefold().strip()
    return ("insufficient_evidence" in text or bool(re.search(
        r"\b(?:manual|documentation|provided (?:evidence|information))\b.{0,65}"
        r"\b(?:does not|doesn't|do not|don't|cannot|can't|fails to|insufficient)\b.{0,65}"
        r"\b(?:provide|mention|describe|explain|specify|contain|cover|show|verify|support|include)\b",
        text,
    )) or bool(re.search(r"\b(?:no mention|not mentioned|not enough information|insufficient evidence|can't verify|cannot verify)\b", text)))
