"""Explicit query/scope contract and lossless provenance for retrieval experiments.

Opt-in: existing chat callers retain their behavior until an adapter supplies this
contract. A resolved question is shared, not independently rebuilt by each stage.
"""
from dataclasses import asdict, dataclass
import hashlib


@dataclass(frozen=True)
class RetrievalRequest:
    request_id: str
    original_question: str
    resolved_question: str
    product_id: str
    versions: tuple[str, ...]
    document_id: str | None = None
    manual_type: str | None = None
    intent: str = 'manual_question'
    history_ids: tuple[str, ...] = ()
    scope_origin: str = 'explicit_question'

    def __post_init__(self):
        if not self.resolved_question.strip() or not self.product_id or not self.versions:
            raise ValueError('Resolved query and explicit product/version scope are required')

    @property
    def query(self):
        return self.resolved_question

    def check(self, query, candidate):
        if query != self.query:
            raise ValueError('Reranker query differs from retrieval request')
        if candidate.get('retrieval_query',self.query)!=self.query or candidate.get('retrieval_request_id',self.request_id)!=self.request_id:
            raise ValueError('Candidate retrieval provenance differs from request')
        if candidate.get('product_id') != self.product_id or str(candidate.get('product_version')) not in self.versions:
            raise ValueError('Candidate outside requested product/version scope')
        if self.document_id and candidate.get('document_id') != self.document_id:
            raise ValueError('Candidate outside requested document scope')

    def to_dict(self):
        return asdict(self)


def document_text(candidate, *, metadata=False):
    body='\n'.join([*candidate['headings'],candidate['text']])
    if not metadata:
        return body
    return ('Product: '+str(candidate.get('product_id','unknown'))+
            '\nManual version: '+str(candidate.get('product_version','unknown'))+
            '\nDocument: '+str(candidate.get('document_id','unknown'))+
            '\nSection and source excerpt:\n'+body)


def trace_excerpt(candidate, source_text, start, end, *, matched_part=None,
                  matched_range=None, block_ids=(), policy='preserve_match'):
    if not 0 <= start < end <= len(source_text):
        raise ValueError('Invalid source character range')
    if candidate['text'] != source_text[start:end]:
        raise ValueError('Excerpt does not equal the recorded source span')
    return dict(source_id=candidate.get('section_id',candidate['chunk_id']),
                source_text_sha256=hashlib.sha256(source_text.encode()).hexdigest(),
                excerpt_sha256=hashlib.sha256(candidate['text'].encode()).hexdigest(),
                source_char_start=start,source_char_end=end,
                matched_part=matched_part,matched_range=matched_range,
                block_ids=list(block_ids),policy=policy)
