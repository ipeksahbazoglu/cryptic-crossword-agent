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
uv run ruff check . && uv run ruff format --check .
uv run mypy src tests                          # strict, with the pydantic plugin
uv run pytest                                  # tests never hit the network or an LLM
uv run pytest tests/test_models.py::test_hidden_word   # single test
uv run pre-commit run --all-files              # hooks call .venv/bin tools, so run `uv sync` first
uv run python -c "import nltk; nltk.download('words')"   # word list for tools/ (integration tests skip without it)
# macOS CERTIFICATE_VERIFY_FAILED on that download? Prefix with: SSL_CERT_FILE=$(uv run python -m certifi)
```

CI (`.github/workflows/ci.yml`) runs `uv sync --locked`, ruff, `mypy src tests`, downloads the NLTK corpus, and runs pytest. The `ci-gate` ruleset protects `main` only: CI must pass, and force pushes and deletions are blocked. Commits follow Conventional Commits. For stacked PRs, merge the lower PR, retarget the next one to `main`, and only then delete the merged branch; deleting it first makes GitHub close the dependent PR.

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
- **`tools/`:** the verification tools. `Dictionary` gives O(1) membership and anagram lookup, indexed by sorted letters, and is passed in rather than kept as a global. `wordplay.py` has `find_anagrams`, `check_hidden_word`, `check_answer` (which checks multi-word answers word by word) and `reverse_letters`. Each returns a frozen pydantic result.
- **`config.py` / `cli.py`:** nothing happens at import time. The CLI loads `.env` from the current directory, and each handler returns an exit code. `extract`, `solve` and `eval` are still stubs.

**Solver design, decided but not yet built:** agentic retrieval. The LLM gets the clue plus tools. Some tools search the clue store: similar definitions and the answers they led to, what an indicator signalled in past clues, how an answer has been clued before. The rest are the verification tools. The model segments the clue in its own reasoning and decides what to look up. An answer is accepted only once a verification tool confirms it, and the answer cites the historical clues it used.

**LLM provider: Groq free tier** (`GROQ_API_KEY` in `.env`). `config.py` and `.env.example` still use Anthropic placeholders until the LLM client is built. Groq has no embeddings API, so retrieval uses a local embedding model. Free-tier rate limits mean LLM jobs need client-side throttling, backoff on 429, and resumable runs. Check which models are available via Groq's `/models` endpoint rather than assuming.
