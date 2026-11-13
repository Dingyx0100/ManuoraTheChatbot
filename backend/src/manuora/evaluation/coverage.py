"""Does a set of evidence passages contain what the gold labels require?

Two gold formats exist in data/evaluation:

* version facts (version_comparison_v0_1): `required_facts` / `accepted_alternatives`,
  each listing quote anchors with a manual `version`. A fact is covered when every
  anchor of one option is quoted verbatim (after normalisation) by some passage.
* span groups (workflow and contextual sets): `span_groups`, each a list of
  alternative {chunk id, quote} anchors; every group must be covered.

`complete` means *all* requirements are covered: the number most experiments report.
"""
from __future__ import annotations

from manuora.evaluation.common import norm


def sid(a):
    return str(a['version']) + ':' + str(a['outline_index'])


def _fact_coverage(q, contains, section=None):
    found = []; sections = []
    for f in q['required_facts']:
        options = [f['evidence']] + [a['evidence'] for a in q['accepted_alternatives'] if a['fact_id'] == f['fact_id']]
        if any(all(contains(a) for a in option) for option in options): found.append(f['fact_id'])
        if section and any(all(section(a) for a in option) for option in options): sections.append(f['fact_id'])
    n = len(q['required_facts'])
    return found, sections, n


def version_coverage(q, evidence):
    """Version facts against *section-level* evidence (section ids from version_alignment)."""
    def allowed(a): return a.get('resolved_section_ids', [sid(a)])
    def contains(a): return any(e['section_id'] in allowed(a) and norm(a['quote']) in norm(e['text']) for e in evidence)
    def section(a): return any(e['section_id'] in allowed(a) for e in evidence)
    found, sections, n = _fact_coverage(q, contains, section)
    return dict(found=found, recall=len(found) / n, complete=len(found) == n, section_recall=len(sections) / n,
                both_versions={'35', '36'} <= {e['product_version'] for e in evidence})


def quote_coverage(q, evidence):
    """Version facts against *any* passages (e.g. production chunks): same version + verbatim quote."""
    texts = [(str(e.get('product_version')), norm(e['text'])) for e in evidence]
    def contains(a): return any(v == str(a['version']) and norm(a['quote']) in t for v, t in texts)
    found, _, n = _fact_coverage(q, contains)
    return dict(found=found, recall=len(found) / n, complete=len(found) == n)


def span_coverage(gold, ev):
    """Span groups: each group needs one anchor whose chunk is present and quotes it."""
    found = [any(any(e['chunk_id'] == a['id'] and norm(a['quote']) in norm(e['text']) for e in ev) for a in group)
             for group in gold['span_groups']]
    return dict(complete=all(found), recall=sum(found) / len(found))


def coverage(c, ev):
    """Gold coverage for a mixed case: version facts, or contextual/standalone span groups."""
    if c['subset'] == 'version': return version_coverage(c['gold'], ev)
    return span_coverage(c, ev)


def labels(c):
    """Bit masks preserve conjunctive anchors and accepted alternative groups."""
    if c['subset'] != 'version':
        gold = set(c['gold']['relevant_chunk_ids']); return [int(e['chunk_id'] in gold) for e in c['candidates']], [[1]]
    anchors = []; options = []
    for fact in c['gold']['required_facts']:
        choices = [fact['evidence']] + [a['evidence'] for a in c['gold']['accepted_alternatives'] if a['fact_id'] == fact['fact_id']]; masks = []
        for choice in choices:
            mask = 0
            for anchor in choice:
                if anchor not in anchors: anchors.append(anchor)
                mask |= 1 << anchors.index(anchor)
            masks.append(mask)
        options.append(masks)
    masks = []
    for e in c['candidates']:
        masks.append(sum(1 << i for i, a in enumerate(anchors)
                         if e['section_id'] in a.get('resolved_section_ids', [sid(a)]) and norm(a['quote']) in norm(e['text'])))
    return masks, options
