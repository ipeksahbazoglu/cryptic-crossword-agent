# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

An agent that solves UK cryptic crossword clues by searching a store of real, already-explained clues scraped from Fifteensquared, then verifying candidate answers mechanically. The owner is building it to learn LLM engineering, CI/CD and git practice. Claude writes the code; the owner makes the design decisions and reviews and merges the PRs.

The design comes from `../CLAUDE_CODE_HANDOFF.md`. Its "Repo state" section is out of date; this file and the code are current. `../cryptic-agent-code/` is an earlier proof of concept built on a flat schema. It is superseded, so use it only as a reference, never as a template.

## Commands

```bash
uv sync                                        # install deps + dev group into .venv
uv run cryptic-agent --help                    # CLI: scrape|extract|solve|eval
uv run cryptic-agent scrape --category guardian/quick-cryptic --pages 2   # -> data/raw/*.jsonl
uv run cryptic-agent solve --clue "Senator arranged crime" --enumeration 7 [--pattern T?E?S?N]
uv sync --group demo                           # then open demos/*.ipynb (kernel: .venv); outputs are stripped on commit
uv run ruff check . && uv run ruff format --check .
uv run mypy src tests                          # strict, with the pydantic plugin
uv run pytest                                  # tests never hit the network or an LLM
uv run pytest tests/test_models.py::test_hidden_word   # single test
uv run pre-commit run --all-files              # hooks call .venv/bin tools, so run `uv sync` first
uv run cryptic-agent ingest                    # download sources (checksum-verified) + build data/lexicon/lexicon.sqlite (~1 min)
```

CI (`.github/workflows/ci.yml`) runs `uv sync --locked`, ruff, `mypy src tests`, runs `cryptic-agent ingest --source ukacd` (CI never downloads the 187 MB dataset, so real-lexicon tests skip there), and runs pytest. The `ci-gate` ruleset protects `main` only: CI must pass, and force pushes and deletions are blocked. Commits follow Conventional Commits. For stacked PRs, merge the lower PR, retarget the next one to `main`, and only then delete the merged branch; deleting it first makes GitHub close the dependent PR.

## Architecture

The pipeline is **scrape → extract → store/index → solve → eval**. Each stage reads the previous stage's output from `data/` (gitignored; override with `CRYPTIC_AGENT_DATA_DIR`).

- **`models.py`: the domain schema that every stage shares.** A `Clue` holds `definitions: list[Definition]` and `wordplay: list[WordplayComponent]`, and each component has an indicator and fodder. Validators encode clue anatomy:
  - a definition sits at the start or end of the clue, or covers the `whole` clue
  - indicator and fodder text are whole words that appear in the clue
  - a double definition has 2+ definitions and no wordplay
  - the answer length matches the enumeration

  `Clue.category` is computed from this structure, not stored. `double_definition`, `cryptic_definition`, `and_lit` and `compound` are clue shapes, not `ComponentType`s. `normalize_answer`, `normalize_phrase` and `enumeration_lengths` are the shared text helpers.
- **`jsonl.py`:** typed `read_jsonl(path, Model)` with `file:line` errors, an atomic `write_jsonl`, and `append_jsonl` for resumable jobs.
- **`scraper/`:** `client.py` is a WordPress REST client that limits itself to 1 request/s (with an injectable clock and sleep), sends a User-Agent, and retries 429/5xx via urllib3 `Retry`. `clean.py` is pure HTML→`RawPost` conversion. `scrape.py` merges posts into the existing file by ID. **Bloggers mark clue parts with formatting.** For example, the Quick Cryptic legend says "the definition is in bold and underlined, the indicator is in red". `clean.py` keeps that formatting as `**bold**`, `<u>…</u>` and `<color=red>…</color>`, which plain markdownify would drop. It records formatting, not meaning: extraction reads each blogger's own legend.
- **`lexicon/`:** third-party reference data, the solver's "experience".
  - `sources.py` pins each source to a URL and SHA-256 and downloads it to `data/lexicon/sources/`. Nothing in `data/` is committed: the repo is public, and ODbL requires any published derived database to be ODbL too.
  - Sources: UKACD (word list, BSD-style), Moby Thesaurus (public domain), and George Ho's cryptics dataset (660k past clues with definitions, charades and indicators; ODbL).
  - `build.py` turns them into one SQLite file. Every mined row keeps `clue_ref` → `clue_refs.source_url`. `store.Lexicon` answers `definition_answers`, `synonyms`, `abbreviations` and `indicator_types` with support counts, and its `exclude_urls` hides given puzzles from every lookup. **Evaluation must use `exclude_urls`**, or the solver can look up the clue under test.
  - About 1,300 UKACD entries lost their accents upstream ('pr\ufffdcis'). They are restored only when exactly one attested word fits the pattern, never guessed, so a repair can never add a non-word.
  - Mined abbreviations need 3+ supporting clues, which filters noise like `a → GR`. `lexicon/data/abbreviations.tsv` is a hand-checked list that always counts (it adds "one → I", which the data misses).
- **`tools/`:** the verification tools. `Dictionary.load()` (UKACD + well-attested past answers + repaired entries; 99.7% of real answers) gives O(1) membership and anagram lookup, indexed by sorted letters, and is passed in rather than kept as a global. `wordplay.py` has `find_anagrams`, `check_hidden_word`, `check_answer` (which checks multi-word answers word by word) and `reverse_letters`. Each returns a frozen pydantic result.
- **`config.py` / `cli.py`:** nothing happens at import time. The CLI loads `.env` from the current directory, and each handler returns an exit code. `extract`, `solve` and `eval` are still stubs.

**`agent/`: the solver**, built so it is never confidently wrong.
  - `tools.py`: `Toolbox` declares each tool once (name, description for the model, JSON parameters, function) over the dictionary and lexicon; `run()` returns compact JSON and turns bad calls into error results the model can read. `exclude_urls` flows into every lexicon lookup.
  - `solver.py`: `Solver.solve()` runs the tool-use loop (at most `max_rounds`), then makes one final call with the strict `WORKSHEET_SCHEMA`. Every thought and tool call is recorded as a `Step`.
  - `worksheet.py`: the model's full parse of the clue (definitions with positions, wordplay steps with indicator, fodder and produced letters, link words). The schema is hand-written for providers' strict mode; a test keeps it in sync with the pydantic model.
  - `verify.py`: code re-checks the worksheet. **confirmed** only if the answer fits the enumeration (and `--pattern`) and is a real word, the definitions sit where claimed, at least one wordplay step is verified mechanically and none fails, every clue word has a role, and the model claimed confirmed. Otherwise **pencilled** (fits, not proven: double definitions, synonym charades) or **unsure** (doesn't fit). Don't loosen these rules without tests: they are what "never confidently wrong" rests on.
  - The human-style design (fast pass of code-generated candidates, crossing letters, stuck checklist) is the plan for what comes next; see `../CLAUDE_CODE_HANDOFF.md` for the original design notes.

**LLM provider: Groq free tier**, model `openai/gpt-oss-120b` (`config.MODEL`; key in `.env` as `GROQ_API_KEY`). `llm/client.py` defines a provider-neutral `LLMClient` protocol (OpenAI-style messages in, `Completion` with text, tool calls, usage and the model's separate `reasoning` out). `GroqClient` implements it: it paces requests with `TokenBudget` using the `x-ratelimit-*` headers (free tier: 8,000 tokens/min, 1,000 requests/day), and the SDK retries 429/5xx. Tests use `ScriptedLLM` or an `httpx.MockTransport`, never the real API. Groq has no embeddings API, so any embeddings must be local.
