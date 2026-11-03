import pytest

from reconrag.config import Settings


@pytest.fixture
def settings() -> Settings:
    return Settings(_env_file=None, ocr_enabled=False)  # type: ignore[call-arg]
