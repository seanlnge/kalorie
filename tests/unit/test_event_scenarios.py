from kalorie.data_grepping.event_scenarios import (
    EventScenarioCatalog,
    OpenAIEventScenarioGenerator,
)


class _FakeCompletions:
    def create(self, **_kwargs):
        class _Message:
            content = """
            {
              "topics": ["Blackwell demand", "AI factory capacity"],
              "analyst_questions": ["How constrained is Blackwell supply?"],
              "management_answers": ["Management may discuss strong AI factory demand."],
              "synthetic_call_snippets": [
                "Blackwell demand remains very strong across data centers."
              ],
              "target_phrase_variants": {
                "blackwell": ["next-generation gpu platform", "blackwell ramp"],
                "data center": ["ai factory capacity"]
              },
              "source_rationales": [
                "Pre-call materials emphasize Blackwell and data center demand."
              ]
            }
            """

        class _Choice:
            message = _Message()

        class _Response:
            choices = [_Choice()]

        return _Response()


class _FencedJsonCompletions:
    def create(self, **_kwargs):
        class _Message:
            content = """
            Here is the event dossier:
            ```json
            {
              "topics": ["Retail media"],
              "analyst_questions": [],
              "management_answers": [],
              "synthetic_call_snippets": [],
              "target_phrase_variants": {"advertising": ["retail media"]},
              "source_rationales": []
            }
            ```
            """

        class _Choice:
            message = _Message()

        class _Response:
            choices = [_Choice()]

        return _Response()


class _FakeChat:
    completions = _FakeCompletions()


class _FakeClient:
    chat = _FakeChat()


class _FencedJsonChat:
    completions = _FencedJsonCompletions()


class _FencedJsonClient:
    chat = _FencedJsonChat()


def test_openai_event_scenario_generator_parses_catalog_json():
    generator = OpenAIEventScenarioGenerator(client=_FakeClient(), model="fake-model")

    catalog = generator.generate(
        event_id="KXEARNINGSMENTIONNVDA-26MAY20",
        company_symbol="NVDA",
        company_name="NVIDIA",
        target_phrases=["blackwell", "data center"],
        material_snippets=["NVIDIA pre-call materials discuss Blackwell demand."],
        max_items=4,
    )

    assert catalog == EventScenarioCatalog(
        event_id="KXEARNINGSMENTIONNVDA-26MAY20",
        company_symbol="NVDA",
        company_name="NVIDIA",
        llm_model="fake-model",
        topics=["Blackwell demand", "AI factory capacity"],
        analyst_questions=["How constrained is Blackwell supply?"],
        management_answers=["Management may discuss strong AI factory demand."],
        synthetic_call_snippets=[
            "Blackwell demand remains very strong across data centers."
        ],
        target_phrase_variants={
            "blackwell": ["next-generation gpu platform", "blackwell ramp"],
            "data center": ["ai factory capacity"],
        },
        source_rationales=[
            "Pre-call materials emphasize Blackwell and data center demand."
        ],
    )


def test_openai_event_scenario_generator_extracts_fenced_json_response():
    generator = OpenAIEventScenarioGenerator(client=_FencedJsonClient(), model="fake-model")

    catalog = generator.generate(
        event_id="WMT-2025-Q2",
        company_symbol="WMT",
        company_name="Walmart",
        target_phrases=["advertising"],
        material_snippets=["Walmart discusses retail media."],
        max_items=4,
    )

    assert catalog.topics == ["Retail media"]
    assert catalog.target_phrase_variants == {"advertising": ["retail media"]}


def test_event_scenario_catalog_rejects_transcript_like_outputs():
    catalog = EventScenarioCatalog(
        event_id="event",
        company_symbol="NVDA",
        company_name="NVIDIA",
        llm_model="fake-model",
        topics=["Blackwell"],
        analyst_questions=[],
        management_answers=["Operator: The transcript begins now."],
        synthetic_call_snippets=[],
        source_rationales=[],
    )

    assert catalog.has_transcript_like_output()
