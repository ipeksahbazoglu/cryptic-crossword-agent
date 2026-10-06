"""The structured answer the model must hand back: a full parse of the clue.

Like a human writing the answer in only once they can explain every word, the
model fills in which words are the definition, which are indicators, which are
fodder and which are link words. `verify.py` then checks that parse with code.
"""

from typing import Any, Literal, get_args

from pydantic import BaseModel, ConfigDict

Mechanism = Literal[
    "anagram",
    "charade",
    "hidden_word",
    "reversal",
    "homophone",
    "container",
    "deletion",
    "acrostic",
    "alternation",
    "double_definition",
    "cryptic_definition",
    "other",
]
MECHANISMS: tuple[str, ...] = get_args(Mechanism)


class WorksheetDefinition(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    text: str  # clue words, exactly as printed
    position: Literal["start", "end", "whole"]


class WorksheetStep(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    mechanism: Mechanism
    indicator: str | None  # clue words signalling the mechanism, or null
    fodder: str  # clue words the mechanism works on, exactly as printed
    produces: str  # letters this step contributes to the answer
    explanation: str


class Worksheet(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    answer: str
    definitions: list[WorksheetDefinition]
    wordplay: list[WorksheetStep]
    link_words: list[str]  # short connecting words with no other role ("for", "in", "to")
    confidence: Literal["confirmed", "pencilled", "unsure"]
    alternatives: list[str]


def _strict(properties: dict[str, Any]) -> dict[str, Any]:
    """Strict JSON-schema objects list every property as required and allow no extras."""
    return {
        "type": "object",
        "properties": properties,
        "required": list(properties),
        "additionalProperties": False,
    }


STRING: dict[str, Any] = {"type": "string"}

# Hand-written rather than generated: providers' strict mode accepts a narrower
# dialect of JSON Schema than pydantic emits. A test checks the two agree.
WORKSHEET_SCHEMA: dict[str, Any] = _strict(
    {
        "answer": STRING,
        "definitions": {
            "type": "array",
            "items": _strict(
                {"text": STRING, "position": {"type": "string", "enum": ["start", "end", "whole"]}}
            ),
        },
        "wordplay": {
            "type": "array",
            "items": _strict(
                {
                    "mechanism": {"type": "string", "enum": list(MECHANISMS)},
                    "indicator": {"type": ["string", "null"]},
                    "fodder": STRING,
                    "produces": STRING,
                    "explanation": STRING,
                }
            ),
        },
        "link_words": {"type": "array", "items": STRING},
        "confidence": {"type": "string", "enum": ["confirmed", "pencilled", "unsure"]},
        "alternatives": {"type": "array", "items": STRING},
    }
)
