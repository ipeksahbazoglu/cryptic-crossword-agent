# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

A RAG/tool-use agent for solving UK cryptic crossword clues, trained on explanations scraped from Fifteensquared. This repo (`cryptic-agent/`) is the uv-packaged project, being built up step by step. `config.py` holds settings (model, data paths, API key); `cli.py` has one subcommand per pipeline stage, and handlers are stubs until each stage is ported.

The working pipeline code lives in the sibling directory `../cryptic-agent-code/`. That directory is not under git and is not packaged: it is a set of standalone scripts with a `requirements.txt`. When porting its code into `src/cryptic_agent/`, it has to meet this repo's stricter tooling (strict mypy, ruff rules below, Python 3.13).

## Commands (this repo)

```bash
uv sync                          # install deps + dev group into .venv
uv run cryptic-agent --help      # CLI: scrape|extract|solve|eval (cryptic_agent.cli:main)
uv run ruff check .              # lint (E, F, I, UP, B, SIM, N; line length 100)
uv run ruff format .             # format (double quotes)
uv run mypy src tests                # strict mode
uv run pytest                    # no tests exist yet
uv run pytest path/to/test_x.py::test_name   # single test
uv run python -c "import nltk; nltk.download('words')"   # required before using the anagram/word tools
```

`pre-commit` is a dev dependency, but there is no `.pre-commit-config.yaml` yet. Commit `uv.lock`.

## Pipeline architecture (from `../cryptic-agent-code/`)

The pipeline has four stages. Each stage reads the previous stage's files from `data/`, which is gitignored and can be regenerated:

1. **scraper/fetch_posts.py** calls the Fifteensquared WordPress REST API (`/wp-json/wp/v2/posts`), not rendered HTML. It first resolves a category slug to an ID. For nested slugs like `guardian/quick-cryptic`, it uses only the last segment. It converts HTML to markdown with bold/italic/underline kept, because these marks carry meaning (bold/underline usually marks the definition, italics the indicator). It rate-limits to 1 request/sec and writes `data/raw/<category>.jsonl`.
2. **extraction/extract_clues.py** calls Claude once per post and forces the `record_clues` tool, whose schema comes from the pydantic `ClueRecord` in `extraction/schema.py`. The output is re-validated against pydantic and written to `data/processed/clues.jsonl`. By design, the prompt skips unclear clues rather than guessing: a bad extraction silently corrupts the few-shot/eval data.
3. **agent/solver.py** runs a manual Claude tool-use loop (at most 8 turns) with the tools in `agent/tools.py`. `TOOL_DEFINITIONS` (JSON schemas) and `TOOL_IMPLEMENTATIONS` (name → function) must stay in sync. Few-shot examples are sampled at random from `clues.jsonl`. This project uses in-context examples, not fine-tuning.
4. **eval/evaluate.py** shuffles the dataset with a seed and splits it into a test set and a separate few-shot pool that never overlaps it. It reports exact-match accuracy overall and per `wordplay_type`, and writes `eval_failures.json` next to the dataset.

Key design points:
- **Extraction and solving are deliberately separate LLM calls.** Extraction aims to be faithful to the source; solving aims to reason well. Don't merge them into one prompt.
- **The solver has to verify answers with tools rather than trust what it generates.** The tools are: dictionary-checked anagrams and hidden words using the NLTK `words` corpus, synonyms from the free Datamuse API, word validity, and reversal.
- **The solver's output format is load-bearing.** The system prompt requires a last line of the form `ANSWER: <word> | TYPE: ... | REASONING: ...`. The eval parses it with `ANSWER:\s*([A-Za-z]+)`, so a change to one needs a matching change to the other.
- `WordplayType` in `schema.py` is the shared taxonomy. The extraction prompt, the solver prompt, and the eval's per-type breakdown all depend on it.
- `agent/tools.py` loads the NLTK corpus at import time and raises `SystemExit` if the corpus is missing.
- The scripts use flat sibling imports (`from schema import ...`, `from tools import ...`), and `evaluate.py` adds `agent/` to `sys.path` by hand. Both will need changing when the code moves into the package.
- The model is set as `MODEL = "claude-sonnet-5"` separately in `extract_clues.py` and `solver.py`.
- `ANTHROPIC_API_KEY` is read from the environment or from `.env` via python-dotenv (see `.env.example`).
