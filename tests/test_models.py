import pytest

from cryptic_agent.models import enumeration_lengths, normalize_answer, normalize_phrase


@pytest.mark.parametrize(
    ("enumeration", "expected"),
    [
        ("7", [7]),
        ("3,4", [3, 4]),
        ("5-3", [5, 3]),
        ("2,3,4", [2, 3, 4]),
        ("(7)", [7]),  # models often pass it with brackets
        ("(3, 4)", [3, 4]),
    ],
)
def test_enumeration_lengths(enumeration: str, expected: list[int]) -> None:
    assert enumeration_lengths(enumeration) == expected


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("treason", "TREASON"),
        ("ice cream", "ICECREAM"),
        ("Jack-in-the-box", "JACKINTHEBOX"),
        ("précis", "PRECIS"),  # accents become plain letters, not dropped
        ("Rösti", "ROSTI"),
        ("Ærø", "AERO"),  # letters Unicode can't decompose need a mapping
        ("Straße", "STRASSE"),
    ],
)
def test_normalize_answer(raw: str, expected: str) -> None:
    assert normalize_answer(raw) == expected


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Senator, arranged!", "senator arranged"),
        ("  Setter's   ruse ", "setters ruse"),
        ("Setter’s ruse", "setters ruse"),
        ("Café crème", "cafe creme"),  # accents kept as letters, not dropped
    ],
)
def test_normalize_phrase(raw: str, expected: str) -> None:
    assert normalize_phrase(raw) == expected
