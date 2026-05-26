from kalorie.domain.models import DocumentChunk, TargetPhrase
from kalorie.ml.labeling import (
    find_exact_mentions,
    find_kalshi_settlement_mentions,
    find_lexical_mentions,
    label_document_chunks,
)


def test_exact_mentions_are_case_insensitive_and_boundary_aware():
    assert find_exact_mentions("Guest Traffic growth of 6.8%", "traffic")
    assert find_exact_mentions("Same Restaurant Sales increased", "same restaurant sales")
    assert find_exact_mentions("Digital Revenue Mix was strong", "digital revenue")
    assert not find_exact_mentions("robotaxi was not discussed", "traffic")
    assert not find_exact_mentions("said momentum improved", "AI")


def test_lexical_mentions_use_aliases_separately_from_exact_matches():
    margin = TargetPhrase(
        phrase="margin",
        normalized_phrase="margin",
        aliases=["restaurant-level profit margin"],
    )
    value = TargetPhrase(
        phrase="value proposition",
        normalized_phrase="value proposition",
        aliases=["compelling value proposition"],
    )
    geopolitical = TargetPhrase(
        phrase="geopolitical uncertainty",
        normalized_phrase="geopolitical uncertainty",
        aliases=[],
    )

    assert find_lexical_mentions("Restaurant-level profit margin expanded.", margin)
    assert find_lexical_mentions("A compelling value proposition helped traffic.", value)
    assert not find_lexical_mentions("No macro commentary.", geopolitical)


def test_label_document_chunks_aggregates_exact_and_lexical_labels():
    chunks = [
        DocumentChunk(
            document_id="doc",
            chunk_index=0,
            text="Guest traffic improved.",
            section="headline",
            token_start=0,
            token_end=3,
        ),
        DocumentChunk(
            document_id="doc",
            chunk_index=1,
            text="Restaurant-level profit margin expanded.",
            section=None,
            token_start=3,
            token_end=7,
        ),
    ]
    targets = [
        TargetPhrase(phrase="traffic", normalized_phrase="traffic"),
        TargetPhrase(
            phrase="margin",
            normalized_phrase="margin",
            aliases=["restaurant-level profit margin"],
        ),
    ]

    labels = {label.target_phrase: label for label in label_document_chunks(chunks, targets)}

    assert labels["traffic"].exact_mentioned is True
    assert labels["traffic"].lexical_mentioned is False
    assert labels["margin"].exact_mentioned is True
    assert labels["margin"].lexical_mentioned is True


def test_kalshi_settlement_mentions_allow_plural_and_possessive_forms_only():
    assert find_kalshi_settlement_mentions("Margins expanded.", "margin")
    assert find_kalshi_settlement_mentions("Margin's expansion helped.", "margin")
    assert find_kalshi_settlement_mentions("Margins' expansion helped.", "margin")
    assert find_kalshi_settlement_mentions("Value propositions improved.", "value proposition")
    assert find_kalshi_settlement_mentions(
        "Value proposition's durability improved.",
        "value proposition",
    )

    assert not find_kalshi_settlement_mentions("Marginal improvement was noted.", "margin")
    assert not find_kalshi_settlement_mentions(
        "Valued proposition language appeared.",
        "value proposition",
    )
    assert not find_kalshi_settlement_mentions("Guiding was discussed.", "guidance")


def test_label_document_chunks_company_entity_scope_filters_analyst_context():
    chunks = [
        DocumentChunk(
            document_id="doc",
            chunk_index=0,
            text=(
                "Operator: Our next question is from Jane Analyst.\n"
                "Analyst: Was guidance weaker than expected?\n"
                "CEO: Guidance remains unchanged."
            ),
            section="qa",
            token_start=0,
            token_end=20,
        ),
    ]
    target = [TargetPhrase(phrase="guidance", normalized_phrase="guidance")]

    labels_all = label_document_chunks(chunks, target, entity_scope="all")
    labels_company = label_document_chunks(chunks, target, entity_scope="company_employee")

    assert labels_all[0].exact_mentioned is True
    assert labels_company[0].exact_mentioned is True
    exact_all = [span for span in labels_all[0].match_spans if span.match_type == "exact"]
    exact_company = [span for span in labels_company[0].match_spans if span.match_type == "exact"]
    assert len(exact_all) > len(exact_company)
