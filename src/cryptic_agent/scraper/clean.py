"""Turn a WordPress post (as JSON from the REST API) into a RawPost.

Pure transformation, no network: easy to test against saved API responses.

Fifteensquared bloggers mark clue parts with formatting, and the conventions
vary by blogger (e.g. "the definition is in bold and underlined, the indicator
is in red"). Plain markdown keeps bold/italic but drops underline and colour,
which live in inline CSS, so we keep those as explicit inline tags:

    <span style="color: red">Cook</span>                  ->  <color=red>Cook</color>
    <span style="text-decoration: underline">cut</span>  ->  <u>cut</u>

We record *formatting*, not our guess at its meaning; the extraction step reads
the blogger's own legend ("the indicator is in red") from the post.
"""

import html
import re
from collections.abc import Mapping
from typing import Any

from bs4 import Tag
from markdownify import MarkdownConverter

from cryptic_agent.models import RawPost


def _parse_style(style: str) -> dict[str, str]:
    """'color: red; text-decoration: underline' -> {'color': 'red', 'text-decoration': ...}."""
    declarations = (d.split(":", 1) for d in style.split(";") if ":" in d)
    return {name.strip().lower(): value.strip().lower() for name, value in declarations}


class _PostConverter(MarkdownConverter):
    """markdownify, plus underline and colour kept as inline tags."""

    def convert_span(self, el: Tag, text: str, parent_tags: set[str]) -> str:
        if not text.strip():
            return text
        style = _parse_style(str(el.get("style", "")))
        if "underline" in style.get("text-decoration", ""):
            text = f"<u>{text}</u>"
        if color := style.get("color"):
            text = f"<color={color}>{text}</color>"
        return text

    def convert_u(self, el: Tag, text: str, parent_tags: set[str]) -> str:
        return f"<u>{text}</u>" if text.strip() else text


def html_to_markdown(content_html: str) -> str:
    converter = _PostConverter(heading_style="ATX", strip=["img", "a", "button"])
    text = converter.convert(content_html)
    text = re.sub(r"[ \t]+\n", "\n", text)  # trailing spaces
    return re.sub(r"\n{3,}", "\n\n", text).strip()  # at most one blank line in a row


def clean_post(raw: Mapping[str, Any]) -> RawPost:
    """Build a RawPost from one item of /wp-json/wp/v2/posts."""
    return RawPost(
        id=raw["id"],
        date=raw["date"],
        url=raw["link"],
        title=html.unescape(raw["title"]["rendered"]),  # "Guardian 29,000 &#8211; Paul"
        content_markdown=html_to_markdown(raw["content"]["rendered"]),
    )
