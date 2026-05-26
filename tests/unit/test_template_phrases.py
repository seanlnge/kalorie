from kalorie.data_grepping.template_phrases import OpenAITemplatePhraseGenerator


class _FakeCompletions:
    def create(self, **_kwargs):
        class _Message:
            content = '{"variants":["Traffic growth","Guest traffic momentum","Traffic growth"]}'

        class _Choice:
            message = _Message()

        class _Response:
            choices = [_Choice()]

        return _Response()


class _FakeChat:
    completions = _FakeCompletions()


class _FakeClient:
    chat = _FakeChat()


def test_openai_template_phrase_generator_parses_json_and_dedupes():
    generator = OpenAITemplatePhraseGenerator(client=_FakeClient(), model="fake-model")

    variants = generator.generate(
        target_phrase="traffic",
        material_snippets=["Demand and traffic trends remained strong."],
        max_variants=5,
    )

    assert variants == ["traffic growth", "guest traffic momentum"]
