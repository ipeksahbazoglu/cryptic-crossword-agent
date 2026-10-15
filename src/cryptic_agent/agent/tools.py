"""The tools the solving agent can call, declared once each.

Every tool is one `Tool`: the name, the description the model reads, the JSON
schema of its arguments, and the Python function that runs it. The model never
runs code itself; it asks for a tool by name and the loop calls `run()`.

Results are kept small on purpose: every token counts against the free tier's
8,000 tokens per minute.
"""

import json
from collections.abc import Callable, Collection
from dataclasses import dataclass
from typing import Any

from cryptic_agent.lexicon.store import Lexicon
from cryptic_agent.tools.dictionary import Dictionary
from cryptic_agent.tools.wordplay import (
    check_answer,
    check_hidden_word,
    find_anagrams,
    reverse_letters,
    select_letters,
)

MAX_LIST = 15  # longest list returned to the model


@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    parameters: dict[str, Any]
    function: Callable[..., Any]

    def spec(self) -> dict[str, Any]:
        """The declaration sent to the model."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


def _params(required: list[str], **properties: dict[str, Any]) -> dict[str, Any]:
    return {"type": "object", "properties": properties, "required": required}


STRING = {"type": "string"}
INTEGER = {"type": "integer"}


@dataclass
class Toolbox:
    """All tools, bound to the dictionary and lexicon they use."""

    dictionary: Dictionary
    lexicon: Lexicon
    exclude_urls: Collection[str] = ()  # hides puzzles from lookups during evaluation

    def __post_init__(self) -> None:
        d, lx = self.dictionary, self.lexicon
        excluded = self.exclude_urls
        self.tools: dict[str, Tool] = {
            t.name: t
            for t in [
                Tool(
                    "find_anagrams",
                    "Real words/phrases that are anagrams of exactly these letters (the "
                    "fodder itself is excluded). Use when an anagram indicator is present.",
                    _params(["letters"], letters=STRING, length=INTEGER),
                    lambda letters, length=None: find_anagrams(d, letters, length).model_dump(),
                ),
                Tool(
                    "check_hidden_word",
                    "Real words of `length` letters hidden in consecutive letters of `text` "
                    "(spaces ignored). Set reversed=true for words hidden backwards.",
                    _params(
                        ["text", "length"],
                        text=STRING,
                        length=INTEGER,
                        reversed={"type": "boolean"},
                    ),
                    lambda text, length, reversed=False: check_hidden_word(
                        d, text[::-1] if reversed else text, length
                    ).model_dump(),
                ),
                Tool(
                    "reverse_letters",
                    "The letters of `text` reversed, for reversal clues.",
                    _params(["text"], text=STRING),
                    lambda text: reverse_letters(text).model_dump(),
                ),
                Tool(
                    "select_letters",
                    "Pick letters from the fodder: first or last letter of each word "
                    "(acrostics, 'initially', 'finally'), or odd/even letters ('oddly', "
                    "'regularly'). Pass only the fodder words, not the indicator.",
                    _params(
                        ["text", "which"],
                        text=STRING,
                        which={"type": "string", "enum": ["first", "last", "odd", "even"]},
                    ),
                    lambda text, which: select_letters(text, which).model_dump(),
                ),
                Tool(
                    "check_answer",
                    "Whether a candidate fits the enumeration and every word is in the "
                    "crossword dictionary. Call this on your final candidate.",
                    _params(["answer", "enumeration"], answer=STRING, enumeration=STRING),
                    lambda answer, enumeration: check_answer(d, answer, enumeration).model_dump(),
                ),
                Tool(
                    "definition_answers",
                    "Answers this exact phrase led to when it was the definition in past "
                    "cryptic clues, with how many clues. Strong evidence for the definition side.",
                    _params(["phrase"], phrase=STRING, length=INTEGER),
                    lambda phrase, length=None: _evidence(
                        lx.definition_answers(phrase, length=length, exclude_urls=excluded)
                    ),
                ),
                Tool(
                    "synonyms",
                    "Thesaurus terms related to a word or phrase, optionally of one length.",
                    _params(["phrase"], phrase=STRING, length=INTEGER),
                    lambda phrase, length=None: {
                        "terms": lx.synonyms(phrase, length=length)[: MAX_LIST * 2]
                    },
                ),
                Tool(
                    "abbreviations",
                    "Letters a clue word or phrase can stand for in crosswords, e.g. "
                    "sailor -> AB, TAR; one -> I, A, AN. For charades and containers.",
                    _params(["phrase"], phrase=STRING),
                    lambda phrase: _evidence(lx.abbreviations(phrase, exclude_urls=excluded)),
                ),
                Tool(
                    "indicator_types",
                    "Which wordplay mechanisms this word or phrase has signalled in past "
                    "clues, and how often. Use to test whether a word is an indicator.",
                    _params(["phrase"], phrase=STRING),
                    lambda phrase: _evidence(lx.indicator_types(phrase, exclude_urls=excluded)),
                ),
            ]
        }

    def specs(self) -> list[dict[str, Any]]:
        return [tool.spec() for tool in self.tools.values()]

    def run(self, name: str, arguments: dict[str, Any]) -> str:
        """Run a tool and return its result as compact JSON for the model.

        Problems become an error result rather than an exception, so the model
        can read what went wrong and try again.
        """
        tool = self.tools.get(name)
        if tool is None:
            result: Any = {"error": f"unknown tool {name!r}; available: {sorted(self.tools)}"}
        elif "_raw" in arguments:
            result = {"error": f"arguments were not valid JSON: {arguments['_raw']!r}"}
        else:
            try:
                result = tool.function(**arguments)
            except Exception as exc:  # the model wrote the arguments: anything can arrive
                result = {"error": f"{type(exc).__name__}: {exc}"}
        return json.dumps(result, separators=(",", ":"))


def _evidence(items: list[Any]) -> dict[str, Any]:
    return {
        "results": [
            {"value": e.value, "clues": e.support} | ({"curated": True} if e.curated else {})
            for e in items[:MAX_LIST]
        ],
        "total": len(items),
    }
