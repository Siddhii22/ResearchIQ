"""Provider-independent LLM interface with Gemini and OpenAI-compatible clients."""

import httpx
from google import genai
from google.genai import types

from app.config import Settings, settings


class LLMConfigurationError(RuntimeError):
    """The LLM service is missing required configuration."""


class LLMRequestError(RuntimeError):
    """The configured LLM endpoint returned an unusable response."""


class LLMAuthenticationError(LLMRequestError):
    """The provider rejected authentication or account access."""


class LLMQuotaError(LLMRequestError):
    """The provider reported a quota or rate-limit failure."""


class LLMModelUnavailableError(LLMRequestError):
    """The configured model is unavailable to the provider account."""


class GeminiLLM:
    """Generate text with the official Google GenAI SDK."""

    def __init__(self, configuration: Settings = settings) -> None:
        self.configuration = configuration
        self._client = None

    def _get_client(self):
        api_key = self.configuration.gemini_api_key
        if not api_key or not api_key.strip():
            raise LLMConfigurationError(
                "GEMINI_API_KEY is not configured; set it in the environment or .env file."
            )
        if self._client is None:
            self._client = genai.Client(api_key=api_key)
        return self._client

    @staticmethod
    def _raise_provider_error(error: Exception) -> None:
        status = getattr(error, "code", None)
        if not isinstance(status, (int, str)):
            status = getattr(error, "status_code", None)
        status_name = str(getattr(error, "status", "")).upper()
        status_value = str(status).upper()

        if status_value in {"401", "403"} or status_name in {
            "UNAUTHENTICATED",
            "PERMISSION_DENIED",
        }:
            raise LLMAuthenticationError(
                "Gemini authentication failed. Check GEMINI_API_KEY and account access."
            ) from error
        if status_value == "429" or status_name in {"RESOURCE_EXHAUSTED", "TOO_MANY_REQUESTS"}:
            raise LLMQuotaError(
                "Gemini quota or rate limit exceeded. Check the project's quota and billing."
            ) from error
        if status_value == "404" or status_name == "NOT_FOUND":
            raise LLMModelUnavailableError(
                "The configured Gemini model is unavailable or not enabled for this project."
            ) from error
        if isinstance(error, (TimeoutError, httpx.TimeoutException)):
            raise LLMRequestError("The Gemini request timed out.") from error
        if isinstance(error, httpx.RequestError):
            raise LLMRequestError("Could not reach the Gemini provider.") from error
        raise LLMRequestError("The Gemini provider request failed.") from error

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        client = self._get_client()
        try:
            response = client.models.generate_content(
                model=self.configuration.gemini_model,
                contents=user_prompt,
                config=types.GenerateContentConfig(
                    system_instruction=system_prompt,
                    temperature=0.2,
                ),
            )
        except Exception as error:
            self._raise_provider_error(error)

        try:
            content = response.text
        except Exception as error:
            raise LLMRequestError("Gemini returned an invalid text-generation response.") from error
        if not isinstance(content, str) or not content.strip():
            raise LLMRequestError("Gemini returned an empty answer.")
        return content.strip()


class OpenAICompatibleLLM:
    """Call a configurable API that implements OpenAI chat completions."""

    def __init__(self, configuration: Settings = settings) -> None:
        self.configuration = configuration

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        api_key = self.configuration.llm_api_key
        if not api_key:
            raise LLMConfigurationError(
                "LLM_API_KEY is not configured; set it in the environment or .env file."
            )

        base_url = self.configuration.llm_api_base_url.rstrip("/")
        endpoint = f"{base_url}/chat/completions"
        try:
            response = httpx.post(
                endpoint,
                headers={"Authorization": f"Bearer {api_key}"},
                json={
                    "model": self.configuration.llm_model,
                    "temperature": 0.2,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                },
                timeout=httpx.Timeout(60.0, connect=10.0),
            )
            response.raise_for_status()
            content = response.json()["choices"][0]["message"]["content"]
        except httpx.HTTPStatusError as error:
            if error.response.status_code in {401, 403}:
                raise LLMAuthenticationError("OpenAI-compatible provider authentication failed.") from error
            if error.response.status_code == 429:
                raise LLMQuotaError("OpenAI-compatible provider quota or rate limit exceeded.") from error
            raise LLMRequestError(
                f"OpenAI-compatible provider returned HTTP {error.response.status_code}."
            ) from error
        except httpx.TimeoutException as error:
            raise LLMRequestError("The OpenAI-compatible provider request timed out.") from error
        except httpx.HTTPError as error:
            raise LLMRequestError("Could not reach the OpenAI-compatible provider.") from error
        except (ValueError, KeyError, IndexError, TypeError) as error:
            raise LLMRequestError("The LLM provider returned an invalid chat-completions response.") from error

        if not isinstance(content, str) or not content.strip():
            raise LLMRequestError("The LLM provider returned an empty answer.")
        return content.strip()


def create_llm_service(configuration: Settings = settings):
    """Create the configured provider behind the shared text-generation interface."""
    provider = configuration.llm_provider.strip().lower()
    if provider == "gemini":
        return GeminiLLM(configuration)
    if provider in {"openai_compatible", "openai-compatible"}:
        return OpenAICompatibleLLM(configuration)
    raise LLMConfigurationError(f"Unsupported LLM_PROVIDER: {configuration.llm_provider!r}.")