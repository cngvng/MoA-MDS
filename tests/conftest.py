import pytest


@pytest.fixture(autouse=True)
def isolate_credentials(monkeypatch):
    """Never load a contributor's .env or send requests with real credentials."""
    from src.llms import factory

    monkeypatch.setattr(factory, "_load_dotenv", lambda path: None)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("HF_DATASETS_OFFLINE", "1")
