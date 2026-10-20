---
name: solve-cryptic
description: Solve a UK cryptic crossword clue with the cryptic-crossword MCP tools. Use when the user gives a cryptic clue to solve.
argument-hint: <clue> (<enumeration>) [crossing letters, e.g. S?U?I]
---

Solve this cryptic crossword clue: $ARGUMENTS

Use the cryptic-crossword MCP tools:

1. Call `fast_pass` with the clue (without the enumeration), the enumeration, and any
   crossing letters as `pattern`.
2. Test your ideas with the other tools: past definitions, synonyms, abbreviations,
   indicators, anagrams, hidden words.
3. Call `verify_parse` with your full parse. Say the answer is confirmed only if
   `verify_parse` returned "confirmed"; otherwise say it is pencilled in and which
   checks code could not make.
