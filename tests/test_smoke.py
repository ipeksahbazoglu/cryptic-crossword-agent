import cryptic_agent


def test_package_imports() -> None:
    assert callable(cryptic_agent.main)
