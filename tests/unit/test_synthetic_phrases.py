from kalorie.ml.synthetic_phrases import generate_synthetic_phrase_candidates


def test_generate_synthetic_phrase_candidates_extracts_large_rules_aligned_terms():
    text = """
    NVIDIA discussed Blackwell demand, data center revenue, gross margin,
    supply chain constraints, and AI factory capacity. Data center revenue
    remained the key investor focus while gaming demand improved.
    """

    candidates = generate_synthetic_phrase_candidates(
        [text],
        seed_phrases=["data center"],
        max_candidates=40,
    )

    assert "data center" in candidates
    assert "blackwell" in candidates
    assert "gross margin" in candidates
    assert "supply chain" in candidates
    assert "ai factory" in candidates
    assert len(candidates) <= 40


def test_generate_synthetic_phrase_candidates_filters_boilerplate_and_short_tokens():
    text = """
    The company will host a conference call. Please see forward looking
    statements and investor relations for details. Tariffs affected China demand.
    """

    candidates = generate_synthetic_phrase_candidates([text], max_candidates=20)

    assert "the" not in candidates
    assert "will" not in candidates
    assert "conference call" not in candidates
    assert "forward looking" not in candidates
    assert "tariffs" in candidates
    assert "china demand" in candidates
