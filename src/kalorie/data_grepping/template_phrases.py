import json
from typing import Any

from openai import OpenAI
from pydantic import BaseModel, ConfigDict, Field

from kalorie.data_cleaning import normalize_and_dedupe_phrases


class TemplatePhraseCatalog(BaseModel):
    model_config = ConfigDict(extra="forbid")

    llm_model: str
    phrase_variants: dict[str, list[str]]


class OpenAITemplatePhraseGenerator:
    def __init__(
        self,
        *,
        api_key: str | None = None,
        client: Any | None = None,
        model: str = "gpt-5.4",
    ) -> None:
        self._client = client or OpenAI(api_key=api_key)
        self._model = model

    def generate(
        self,
        *,
        target_phrase: str,
        material_snippets: list[str],
        max_variants: int = 12,
    ) -> list[str]:
        prompt = _template_prompt(
            target_phrase=target_phrase,
            material_snippets=material_snippets,
            max_variants=max_variants,
        )
        response = self._client.chat.completions.create(
            model=self._model,
            temperature=0.2,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
        )
        content = response.choices[0].message.content or ""
        try:
            payload = json.loads(content)
            variants = payload.get("variants", [])
            if not isinstance(variants, list):
                return []
            return normalize_and_dedupe_phrases([str(value) for value in variants])[:max_variants]
        except json.JSONDecodeError:
            lines = [line.strip("- ").strip() for line in content.splitlines() if line.strip()]
            return normalize_and_dedupe_phrases(lines)[:max_variants]

    @property
    def model(self) -> str:
        return self._model


class _PromptPayload(BaseModel):
    target_phrase: str
    max_variants: int = Field(ge=1, le=50)
    snippets: list[str]


def _template_prompt(
    *,
    target_phrase: str,
    material_snippets: list[str],
    max_variants: int,
) -> str:
    payload = _PromptPayload(
        target_phrase=target_phrase,
        max_variants=max_variants,
        snippets=material_snippets,
    )
    instruction = (
        "Given these pre-call supplemental snippets, generate phrase variants that management "
        "might use on the earnings call while expressing the same concept.\n"
    )
    return (
        instruction
        + "Output strict JSON with a single key `variants` (array of strings). "
        + "Do not include commentary.\n"
        + payload.model_dump_json(indent=2)
    )


_SYSTEM_PROMPT = (
    "You generate concise, realistic earnings-call wording variants. "
    "Do not invent facts, numbers, or entities not grounded in the snippets."
)
