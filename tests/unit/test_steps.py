from manuora.retrieval.steps import decompose_prompt, merge_steps, parse_queries


def hits(*ids):
    return [{"chunk_id": i} for i in ids]


def test_parse_queries_accepts_json_and_rejects_junk():
    assert parse_queries('Sure: {"queries": ["a", " b ", ""]}') == ["a", "b"]
    assert parse_queries("no json here") == []
    assert parse_queries('{"queries": "not a list"}') == []
    assert len(parse_queries('{"queries": ["1","2","3","4","5","6"]}')) == 5


def test_decompose_prompt_drops_unversioned():
    prompt = decompose_prompt("How?", "freecad", ["None"])
    assert "freecad  manual" in prompt and "Question: How?" in prompt


def test_merge_takes_each_steps_best_unused_passage_in_step_order():
    question = hits("q1", "s1", "q2")
    steps = [hits("q1", "s1"), hits("s1", "s2"), hits("s3")]
    # step 1's best unused is s1 (q1 taken), step 2's is s2 (s1 taken), step 3 gives s3, then the question's rest
    assert [h["chunk_id"] for h in merge_steps(question, steps)] == ["q1", "s1", "s2", "s3", "q2"]


def test_merge_respects_cap_and_per_step():
    merged = merge_steps(hits("q1"), [hits("a", "b"), hits("c", "d")], per_step=2, max_passages=4)
    assert [h["chunk_id"] for h in merged] == ["q1", "a", "c", "b"]
