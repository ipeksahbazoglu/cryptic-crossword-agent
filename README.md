# cryptic-crossword-agent

An agent that solves UK cryptic crossword clues, and refuses to call an answer
"confirmed" unless code has checked it.

```
$ cryptic-agent solve --clue "Mountain observed in severe storm" --enumeration 7

ANSWER: EVEREST  [CONFIRMED]  (tier 0: code only)
  definition: 'Mountain' (start)
  wordplay:   hidden_word of 'severe storm', indicator 'observed in' -> EVEREST
checks:
  [  ok] answer fits: 7 letters for (7)
  [  ok] real word: in the dictionary
  [  ok] definition 'Mountain': at the start
  [  ok] wordplay: hidden_word of 'severe storm': hidden there
  [  ok] every word has a role: all words accounted for
cost: 0 requests, 0 tokens
```

## How it works

A cryptic clue has two halves that lead to the same answer: a **definition** at
one end, and **wordplay** (an anagram, a hidden word, first letters…) made from
the other words. The solver works like a careful human:

1. **Fast pass (code, no AI).** Looks for quick wins: anagrams of neighbouring
   words, hidden words, first letters, and what each end of the clue has meant in
   660,000 past clues.
2. **Tier 0: code only.** If both halves are found and agree, code writes the
   whole explanation. No AI call, no cost.
3. **Tier 1: one AI call.** If code found the wordplay but not the definition,
   the model is asked once to finish the parse.
4. **Tier 2: the agent.** Otherwise the model reasons with tools (anagram and
   hidden-word checks, past definitions, abbreviations, indicators).
5. **Verify (code, always).** Whatever tier produced the answer, code checks it:
   right length, a real word, the definition really at one end, the wordplay
   really works, every clue word used exactly once. If the same wordplay could
   have made a different word, the definition must be backed by past clues or a
   thesaurus.

The verdict is **confirmed** (every check passed), **pencilled** (it fits, but
something could not be proven) or **unsure**. The model's own confidence never
decides it.

## Setup

Needs Python 3.13 and [uv](https://docs.astral.sh/uv/).

```bash
uv sync                        # install
uv run cryptic-agent ingest    # download the word list and build the lexicon (~1 min, ~500 MB)
echo "GROQ_API_KEY=gsk_..." > .env   # a free key from https://console.groq.com/keys
```

## Use

```bash
# solve one clue; --tokens shows what each AI call cost
uv run cryptic-agent solve --clue "Cook using mere sweet desserts" --enumeration 9 --tokens

# with letters from crossing answers
uv run cryptic-agent solve --clue "Mountain observed in severe storm" --enumeration 7 --pattern "E?E?E?T"

# download explained puzzles from Fifteensquared (test data; no AI involved)
uv run cryptic-agent scrape --category guardian --pages 70

# measure the solver on real clues it has not seen
uv run cryptic-agent eval --n 30 --seed 42 --since 2023-01-01
```

Three notebooks in `demos/` walk through it (run `uv sync --group demo` first):
the data, the agent solving clues step by step, and where the tokens go.

## Measuring it honestly

`eval` solves a fixed, reproducible sample of clues with known answers, spread
across clue types. Each clue's own explanation is hidden from the solver's
memory, so it cannot look the answer up. The report gives accuracy by clue type
and by tier, tokens per clue, and the number that matters most: **confirmed
answers that were wrong**, which must stay at zero.

Results so far are from small samples, so treat them as early signs, not
benchmarks: about 13 of 15 Guardian *Quick Cryptic* clues right, and 3 of 6 full
Guardian cryptics, with no confirmed answer wrong in either.

## Limits

- **Free-tier quota.** The model runs on Groq's free tier: 8,000 tokens a minute
  and 200,000 a day. A hard clue can use over 20,000 tokens, so only a few dozen
  solves fit in a day.
- **Not every mechanism can be checked by code yet.** Charades, containers,
  deletions, homophones and double definitions are solved but stay "pencilled".
- **Older blog layouts.** The parser reads about 90% of Guardian posts from 2014
  on; earlier posts use layouts it does not handle.

## Data and credits

Nothing below is stored in this repository; `ingest` and `scrape` download it to
a local `data/` folder.

- **[UKACD](https://pypi.org/project/ccxxv/)**, the UK Advanced Cryptics
  Dictionary, © 2009 J Ross Beresford (BSD-style licence).
- **[Moby Thesaurus II](https://www.gutenberg.org/ebooks/3202)** by Grady Ward
  (public domain).
- **[cryptics.georgeho.org](https://cryptics.georgeho.org/)**, George Ho's
  dataset of past cryptic clues (Open Database License).
- Clue explanations for testing come from the bloggers of
  **[Fifteensquared](https://fifteensquared.net)**, fetched politely through the
  site's public API.

## Development

```bash
uv run pytest                              # tests never call the network or the AI
uv run ruff check . && uv run mypy src tests
uv run pre-commit install
```

## License

MIT, see [LICENSE](LICENSE). The data sources above have their own licences.
