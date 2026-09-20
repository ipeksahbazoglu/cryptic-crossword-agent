# cryptic-crossword-agent

A RAG-based agent for solving UK cryptic crossword clues.

## Status

Early development — not yet functional.

## Plan

- Scrape clue/answer/explanation data from Fifteensquared
- Decompose clues into definition, indicator, and fodder components
- Build a retrieval pipeline: match a new clue's components against
  historical ones to inform candidate answers
- Verify candidates mechanically (anagram/hidden-word/etc. checks)
  rather than trusting LLM generation alone

## License

See LICENSE.