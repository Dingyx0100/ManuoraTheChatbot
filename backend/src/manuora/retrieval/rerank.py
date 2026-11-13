"""Qwen3 cross-encoder reranking of fused source chunks."""

from __future__ import annotations

from .request import RetrievalRequest, document_text


class QwenReranker:
    def __init__(self, model_id: str, revision: str, *, device: str = "cpu",
                 max_length: int = 768):
        from sentence_transformers import CrossEncoder

        self.model = CrossEncoder(model_id, revision=revision, device=device,
                                  max_length=max_length, trust_remote_code=False,
                                  local_files_only=True)

    def rerank(self, query: str, candidates: list[dict], *, limit: int = 5,
               input_limit: int = 12, request: RetrievalRequest | None = None,
               metadata: bool = False) -> list[dict]:
        selected = candidates[:input_limit]
        if not selected:
            return []
        if request:
            for candidate in selected:
                request.check(query,candidate)
        pairs = [(query, document_text(candidate,metadata=metadata)) for candidate in selected]
        scores = self.model.predict(pairs, batch_size=2, show_progress_bar=False)
        ranked = [{**candidate, "rerank_score": float(score)}
                  for candidate, score in zip(selected, scores)]
        return sorted(ranked, key=lambda hit: hit["rerank_score"], reverse=True)[:limit]
