# %% [markdown]
# # Demo 1: scraping Fifteensquared and parsing clues
#
# Walks through the pipeline built so far, using the real code in `src/cryptic_agent`:
#
# 1. talk to the Fifteensquared WordPress API
# 2. see how cleaning keeps the blogger's formatting (red indicator, underlined definition)
# 3. scrape a whole category into `data/raw/`
# 4. parse every post's clue table into structured clues, with no LLM
# 5. look at one clue up close, and at the dataset as a whole
# 6. run the verification tools on real clues
#
# **Run it** either way:
# - VS Code: open this file and click "Run Cell" above each `# %%` (Interactive Window)
# - terminal: `uv sync --group demo && uv run python demos/01_scrape_and_parse.py`
#
# Cells 1 and 3 make a handful of live requests (rate-limited to 1/second).

# %%
import os
import re
from collections import Counter
from pathlib import Path

import pandas as pd
from IPython.display import display


def find_repo_root(start: Path) -> Path:
    """The folder with pyproject.toml. VS Code runs cells from demos/, not the repo root."""
    for folder in (start, *start.parents):
        if (folder / "pyproject.toml").exists():
            return folder
    raise FileNotFoundError("run this from inside the cryptic-agent repo")


ROOT = find_repo_root(Path.cwd())
os.environ.setdefault("CRYPTIC_AGENT_DATA_DIR", str(ROOT / "data"))

from cryptic_agent import config  # noqa: E402  (after setting the data dir)
from cryptic_agent.extraction.table_parser import ParsedClue, parse_clue_table  # noqa: E402
from cryptic_agent.jsonl import read_jsonl  # noqa: E402
from cryptic_agent.models import RawPost  # noqa: E402
from cryptic_agent.scraper.clean import clean_post  # noqa: E402
from cryptic_agent.scraper.client import WordPressClient  # noqa: E402
from cryptic_agent.scraper.scrape import output_path, scrape_category  # noqa: E402

pd.set_option("display.max_colwidth", 45)
pd.set_option("display.max_columns", None)  # show every column, also when run as a script
pd.set_option("display.width", 250)
CATEGORY = "guardian/quick-cryptic"
print("data dir:", config.data_dir())

# %% [markdown]
# ## 1. Talk to the API
# The scraper uses the site's public JSON API (`/wp-json/wp/v2`), not its web pages.
# Category slugs are resolved to numeric IDs first, then posts are fetched newest first.

# %%
client = WordPressClient()
category_id = client.resolve_category_id(CATEGORY)
print(f"{CATEGORY!r} -> category id {category_id}")

latest = list(client.iter_posts(category_id, max_pages=1, per_page=3))
for raw in latest:
    print(f"  {raw['date'][:10]}  {raw['title']['rendered']}")

# %% [markdown]
# ## 2. Before and after cleaning
# Bloggers mark clue parts with inline CSS. Plain HTML-to-markdown conversion throws that
# away; ours keeps it as `<color=red>…</color>` and `<u>…</u>`.

# %%
html = latest[0]["content"]["rendered"]
row = next(r for r in re.findall(r"<tr>.*?</tr>", html, re.S) if "color: red" in r)
clue_html = re.search(r"<div>(.*?)</div>", row, re.S)
before = clue_html.group(1).strip() if clue_html else row

print("RAW HTML:\n ", before)
print(
    "\nCLEANED: \n ",
    next(line for line in clean_post(latest[0]).content_markdown.splitlines() if "<color=" in line),
)

# %% [markdown]
# ## 3. Scrape the whole category
# Two pages of 100 posts. Re-running merges by post ID, so a second run reports "0 new".

# %%
out = output_path(config.raw_dir(), CATEGORY)
summary = scrape_category(client, CATEGORY, max_pages=2, out_path=out)
print(f"fetched {summary.fetched} posts, {summary.new} new, {summary.total} in {summary.path}")

posts = read_jsonl(out, RawPost)
print(f"{len(posts)} posts from {posts[-1].date:%Y-%m-%d} to {posts[0].date:%Y-%m-%d}")

# %% [markdown]
# ## 4. Parse every post's clue table (no LLM)
# Each table row becomes a `ParsedClue`: the definition comes from the underlining, the
# indicator from the red text, and type hints from the blogger's italics.

# %%
clues: list[ParsedClue] = [clue for post in posts for clue in parse_clue_table(post)]
df = pd.DataFrame([c.model_dump() for c in clues])
print(f"{len(clues)} clues from {df.source_title.nunique()} posts")

display(
    df[["clue_text", "enumeration", "answer", "definitions", "indicators", "type_hints"]].head(12)
)

# %% [markdown]
# ## 5. One clue up close
# Change the answer to look at any other clue.

# %%
by_answer = {c.answer: c for c in clues}


def show(answer: str) -> None:
    clue = by_answer[answer.upper()]
    print(f"{clue.source_title}, {clue.number} {clue.direction}\n")
    print("as scraped :", clue.clue_markup, f"({clue.enumeration})")
    print("clue       :", clue.clue_text, f"({clue.enumeration})")
    print("answer     :", clue.answer)
    print("definitions:", clue.definitions)
    print("indicators :", clue.indicators)
    print("type hints :", clue.type_hints)
    print("parsing    :", clue.parsing)


show("MERINGUES")

# %% [markdown]
# ## 6. The dataset as a whole


# %%
def layout(post: RawPost) -> str:
    text = post.content_markdown
    if "| Answer " in text:
        return "2025+: 'Answer X' + parsing row"
    if re.search(r"^\| \d+ \| [A-Z][A-Z' -]+ \|", text, re.M):
        return "early 2024: answer first"
    return "late 2024: bare answer"


layouts = {p.url: layout(p) for p in posts}
df["layout"] = df.source_url.map(layouts)
df["n_definitions"] = df.definitions.map(len)
df["has_indicator"] = df.indicators.map(bool)

print("clues per table layout:")
display(df.layout.value_counts())

print("\nwordplay type hints (from the blogger's italics):")
display(df.type_hints.explode().value_counts(dropna=False))

print("\ndefinitions per clue (2 = likely double definition, 0 = cryptic definition or unmarked):")
display(df.n_definitions.value_counts().sort_index())

print("\nshare of clues with a marked indicator, by layout:")
display(df.groupby("layout").has_indicator.mean().round(2))

# %% [markdown]
# ## 7. Verification tools on real clues
# These are the checks the solver will use to confirm an answer instead of trusting a guess.

# %%
from cryptic_agent.models import normalize_answer  # noqa: E402
from cryptic_agent.tools.dictionary import Dictionary, DictionaryNotFoundError  # noqa: E402
from cryptic_agent.tools.wordplay import (  # noqa: E402
    check_answer,
    check_hidden_word,
    find_anagrams,
)

try:
    dictionary = Dictionary.load()  # UKACD + lexicon words; run `cryptic-agent ingest` first
except DictionaryNotFoundError as exc:
    raise SystemExit(str(exc)) from exc
print(f"dictionary: {len(dictionary):,} words")

# %% [markdown]
# **Hidden words.** For every clue the blogger called a hidden word, is the answer
# really sitting in the clue text, and does `check_hidden_word` find it?

# %%
hidden = [c for c in clues if "hidden_word" in c.type_hints]
in_text = found = 0
for c in hidden:
    squashed = normalize_answer(c.clue_text)
    in_text += c.answer in squashed or c.answer[::-1] in squashed
    matches = check_hidden_word(dictionary, c.clue_text, len(c.answer)).matches
    found += c.answer in {m.word for m in matches}

print(f"{len(hidden)} hidden-word clues")
print(f"  answer spelled out in the clue (forwards or backwards): {in_text}")
print(f"  found by check_hidden_word (forwards, in dictionary):   {found}")

example = hidden[0]
result = check_hidden_word(dictionary, example.clue_text, len(example.answer))
print(f"\ne.g. {example.clue_text!r} -> {[m.word for m in result.matches]}")

# %% [markdown]
# **Anagrams.** The fodder isn't parsed yet (that's the LLM's job in step F3), but we can
# search the clue for a run of consecutive words whose letters rearrange into the answer.


# %%
def find_fodder(clue: ParsedClue) -> str | None:
    """The shortest run of clue words whose letters are an anagram of the answer."""
    words = clue.clue_text.split()
    target = sorted(clue.answer)
    for size in range(1, len(words) + 1):
        for start in range(len(words) - size + 1):
            run = " ".join(words[start : start + size])
            if sorted(normalize_answer(run)) == target:
                return run
    return None


anagrams = [c for c in clues if c.type_hints == ["anagram"]]
with_fodder = [(c, f) for c in anagrams if (f := find_fodder(c))]
print(f"{len(anagrams)} anagram clues; plain-word fodder found in {len(with_fodder)}")
print("(the rest use abbreviations or extra letters, e.g. 'one' = I, 'about' = C)\n")

for clue, fodder in with_fodder[:6]:
    result = find_anagrams(dictionary, fodder)
    verdict = "✓" if clue.answer in result.matches else "✗ (answer not in dictionary)"
    print(f"{clue.clue_text!r:60} fodder {fodder!r:22} -> {clue.answer} {verdict}")

# %% [markdown]
# **Dictionary coverage.** How many answers does the dictionary accept? It is UKACD plus
# well-attested past answers and repaired UKACD entries (81% with NLTK, where we started).
# Multi-word answers are checked word by word, e.g. ICE CREAM (3,5).

# %%
checks = [check_answer(dictionary, c.answer, c.enumeration) for c in clues]
accepted = sum(r.valid for r in checks)
print(f"{accepted}/{len(checks)} answers accepted ({accepted / len(checks):.1%})")

unknown = Counter(word for r in checks for word in r.unknown_words)
print("\nwords still unknown:", [w for w, _ in unknown.most_common(15)])

# %% [markdown]
# ## 8. Memory: the lexicon
# What an experienced solver "just knows", looked up in `data/lexicon/lexicon.sqlite`
# (built by `cryptic-agent ingest` from 660k past clues, the Moby thesaurus and a curated
# abbreviation list). Every answer comes with how many past clues back it.

# %%
from cryptic_agent.lexicon.store import Evidence, Lexicon  # noqa: E402

lexicon = Lexicon()


def evidence(items: list[Evidence], k: int = 6) -> str:
    return ", ".join(f"{e.value} ({e.support}{'*' if e.curated else ''})" for e in items[:k])


print(
    "definition 'Love god' (4)     ->", evidence(lexicon.definition_answers("Love god", length=4))
)
print(
    "definition 'Prime Minister' (6)->",
    evidence(lexicon.definition_answers("Prime Minister", length=6)),
)
print("thesaurus 'joyful' (7)         ->", lexicon.synonyms("joyful", length=7)[:8])
print("abbreviation 'sailor'          ->", evidence(lexicon.abbreviations("sailor")))
print("abbreviation 'one'             ->", evidence(lexicon.abbreviations("one")), " (* = curated)")
print("indicator 'about'              ->", evidence(lexicon.indicator_types("about")))
print("indicator 'cook'               ->", evidence(lexicon.indicator_types("cook")))

# %% [markdown]
# **How often does the definition alone lead to the answer?** For each Quick Cryptic clue
# with one marked definition: is the answer among past answers for that definition, or
# thesaurus terms, of the right length? And how many candidates would the solver face?

# %%
single = [c for c in clues if len(c.definitions) == 1]
found, shortlist = 0, []
for c in single:
    length = len(c.answer)
    candidates = {e.value for e in lexicon.definition_answers(c.definitions[0], length=length)}
    candidates |= set(lexicon.synonyms(c.definitions[0], length=length))
    if c.answer in candidates:
        found += 1
        shortlist.append(len(candidates))

shortlist.sort()
print(
    f"{found}/{len(single)} clues ({found / len(single):.0%}): answer reachable from the definition"
)
print(f"candidates to choose from when it is: median {shortlist[len(shortlist) // 2]}")
print("The wordplay check then picks between them; the rest is the LLM's job.")
