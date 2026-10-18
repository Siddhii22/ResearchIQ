"""Mocked provider tests for the provider-independent LLM service."""

from types import SimpleNamespace
from unittest.mock import Mock

import httpx
import pytest

from app.config import Settings
from app.services.llm_service import (
    GeminiLLM,
    LLMAuthenticationError,
    LLMConfigurationError,
    LLMModelUnavailableError,
    LLMProviderUnavailableError,
    LLMQuotaError,
    LLMRequestError,
    LLMTimeoutError,
    OpenAICompatibleLLM,
    create_llm_service,
)


def gemini_settings(**changes: object) -> Settings:
    values: dict[str, object] = {
        "llm_provider": "gemini",
        "gemini_api_key": "mocked-test-key",
        "gemini_model": "gemini-3.8-flash",
    }
    values.update(changes)
    return Settings(**values)


def test_gemini_uses_official_sdk_and_shared_string_interface(monkeypatch: pytest.MonkeyPatch) -> None:
    response = SimpleNamespace(text=" Generated text. ")
    models = Mock()
    models.generate_content.return_value = response
    client = SimpleNamespace(models=models)
    client_factory = Mock(return_value=client)
    monkeypatch.setattr("app.services.llm_service.genai.Client", client_factory)
    llm = GeminiLLM(gemini_settings())

    result = llm.generate("Evidence-only system prompt", "Research question")

    assert result == "Generated text."
    client_factory.assert_called_once()
    http_options = client_factory.call_args.kwargs["http_options"]
    assert http_options.timeout == 30_000
    assert http_options.retry_options.attempts == 1
    models.generate_content.assert_called_once()
    arguments = models.generate_content.call_args.kwargs
    assert arguments["model"] == "gemini-3.8-flash"
    assert arguments["contents"] == "Research question"
    assert arguments["config"].system_instruction == "Evidence-only system prompt"
    assert arguments["config"].temperature == 0.2


def test_gemini_client_is_reused(monkeypatch: pytest.MonkeyPatch) -> None:
    models = Mock()
    models.generate_content.return_value = SimpleNamespace(text="ok")
    client_factory = Mock(return_value=SimpleNamespace(models=models))
    monkeypatch.setattr("app.services.llm_service.genai.Client", client_factory)
    llm = GeminiLLM(gemini_settings())

    assert llm.generate("system", "first") == "ok"
    assert llm.generate("system", "second") == "ok"

    client_factory.assert_called_once()
    assert models.generate_content.call_count == 2


def test_gemini_requires_key_before_client_creation(monkeypatch: pytest.MonkeyPatch) -> None:
    client_factory = Mock()
    monkeypatch.setattr("app.services.llm_service.genai.Client", client_factory)
    llm = GeminiLLM(gemini_settings(gemini_api_key="  "))

    with pytest.raises(LLMConfigurationError, match="GEMINI_API_KEY is not configured"):
        llm.generate("system", "user")
    client_factory.assert_not_called()


@pytest.mark.parametrize(
    ("status", "exception_type", "message"),
    [
        (401, LLMAuthenticationError, "Gemini authentication failed"),
        (403, LLMAuthenticationError, "Gemini authentication failed"),
        (429, LLMQuotaError, "quota or rate limit exceeded"),
        (404, LLMModelUnavailableError, "model is unavailable"),
        (503, LLMProviderUnavailableError, "Gemini is temporarily unavailable"),
    ],
)
def test_gemini_maps_provider_errors_without_leaking_details(
    monkeypatch: pytest.MonkeyPatch,
    status: int,
    exception_type: type[Exception],
    message: str,
) -> None:
    secret_like_provider_text = "sensitive-provider-detail"
    provider_error = RuntimeError(secret_like_provider_text)
    provider_error.code = status
    models = Mock()
    models.generate_content.side_effect = provider_error
    monkeypatch.setattr(
        "app.services.llm_service.genai.Client",
        Mock(return_value=SimpleNamespace(models=models)),
    )
    llm = GeminiLLM(gemini_settings())

    with pytest.raises(exception_type, match=message) as error:
        llm.generate("system", "user")
    assert secret_like_provider_text not in str(error.value)


def test_gemini_does_not_retry_503(monkeypatch: pytest.MonkeyPatch) -> None:
    provider_error = RuntimeError("temporary failure")
    provider_error.code = 503
    models = Mock()
    models.generate_content.side_effect = provider_error
    monkeypatch.setattr(
        "app.services.llm_service.genai.Client",
        Mock(return_value=SimpleNamespace(models=models)),
    )

    with pytest.raises(LLMProviderUnavailableError) as error:
        GeminiLLM(gemini_settings()).generate("system", "user")

    models.generate_content.assert_called_once()
    assert error.value.__cause__ is provider_error
    assert "temporary failure" not in str(error.value)


def test_gemini_does_not_retry_429(monkeypatch: pytest.MonkeyPatch) -> None:
    provider_error = RuntimeError("quota details")
    provider_error.code = 429
    models = Mock()
    models.generate_content.side_effect = provider_error
    monkeypatch.setattr(
        "app.services.llm_service.genai.Client",
        Mock(return_value=SimpleNamespace(models=models)),
    )
    with pytest.raises(LLMQuotaError):
        GeminiLLM(gemini_settings()).generate("system", "user")

    models.generate_content.assert_called_once()


def test_gemini_empty_text_is_an_explicit_provider_error(monkeypatch: pytest.MonkeyPatch) -> None:
    models = Mock()
    models.generate_content.return_value = SimpleNamespace(text=" ")
    monkeypatch.setattr(
        "app.services.llm_service.genai.Client",
        Mock(return_value=SimpleNamespace(models=models)),
    )

    with pytest.raises(LLMRequestError, match="Gemini returned an empty answer"):
        GeminiLLM(gemini_settings()).generate("system", "user")


def test_gemini_timeout_maps_to_timeout_error(monkeypatch: pytest.MonkeyPatch) -> None:
    models = Mock()
    models.generate_content.side_effect = httpx.ReadTimeout("provider timeout")
    monkeypatch.setattr(
        "app.services.llm_service.genai.Client",
        Mock(return_value=SimpleNamespace(models=models)),
    )

    with pytest.raises(LLMTimeoutError, match="Gemini request timed out"):
        GeminiLLM(gemini_settings()).generate("system", "user")


def test_provider_factory_selects_gemini_and_openai_compatible() -> None:
    assert isinstance(create_llm_service(gemini_settings()), GeminiLLM)
    assert isinstance(
        create_llm_service(
            Settings(llm_provider="openai_compatible", llm_api_key="mocked", llm_model="gpt-test")
        ),
        OpenAICompatibleLLM,
    )


def test_openai_compatible_provider_remains_supported(monkeypatch: pytest.MonkeyPatch) -> None:
    response = Mock()
    response.json.return_value = {"choices": [{"message": {"content": " OpenAI answer "}}]}
    monkeypatch.setattr("app.services.llm_service.httpx.post", Mock(return_value=response))
    llm = OpenAICompatibleLLM(
        Settings(
            llm_api_key="mocked-test-key",
            llm_api_base_url="https://example.test/v1",
            llm_model="test-model",
        )
    )

    assert llm.generate("system", "user") == "OpenAI answer"


def test_missing_gemini_configuration_fails_clearly() -> None:
    with pytest.raises(LLMConfigurationError, match="GEMINI_API_KEY is not configured"):
        GeminiLLM(gemini_settings(gemini_api_key=None)).generate("system", "user")