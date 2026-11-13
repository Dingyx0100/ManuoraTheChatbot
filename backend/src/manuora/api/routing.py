"""Small, conservative routes for requests that do not need manual retrieval."""

from __future__ import annotations

import re


def early_response(question: str, assistant_name: str, product_ids: set[str]) -> str | None:
    text = re.sub(r"[?!.\s]+$", "", question.casefold().strip())
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

    if (re.search(r"\b(latest|newest|current)\b", text)
            and re.search(r"\b(version|release)\b", text)
            and re.search(r"\b(what|which|is)\b", text)
            and not re.search(r"\b(update|upgrade|install|configure|support|supported|compatible)\b", text)):
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


def resolve_scope(question: str, documents: list[dict], product_id: str | None,
                  product_version: str | None) -> tuple[str | None, str | None, str | None]:
    """Use an explicit product/version in the question unless it conflicts with the UI."""
    text = question.casefold()
    mentioned = [product for product in {row["product_id"] for row in documents}
                 if re.search(rf"\b{re.escape(product.casefold())}\b", text)]
    if len(mentioned) != 1:
        return product_id, product_version, None
    product = mentioned[0]
    if product_id and product_id != product:
        return product_id, product_version, "The selected product differs from the one in your question. Please choose which one you mean."
    versions = {str(row["product_version"]) for row in documents
                if row["product_id"] == product and row["product_version"]}
    explicit = [version for version in versions if re.search(
        rf"\b{re.escape(product.casefold())}\s+v?{re.escape(version.casefold())}\b", text)]
    if len(explicit) == 1:
        if product_version and product_version != explicit[0]:
            return product, product_version, "The selected version differs from the one in your question. Please choose which version you mean."
        return product, explicit[0], None
    return product, product_version, None
