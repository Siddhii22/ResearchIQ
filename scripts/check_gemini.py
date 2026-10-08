"""Run one sanitized Gemini configuration and generation smoke check."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.config import settings
from app.services.llm_service import (
    GeminiLLM,
    LLMAuthenticationError,
    LLMConfigurationError,
    LLMModelUnavailableError,
    LLMQuotaError,
    LLMRequestError,
)


def main() -> int:
    key_present = bool(settings.gemini_api_key and settings.gemini_api_key.strip())
    print(f"GEMINI_API_KEY present: {'YES' if key_present else 'NO'}")
    if not key_present:
        print("Configured model reachable: UNKNOWN")
        print("Diagnostic result: missing configuration")
        return 1

    try:
        GeminiLLM(settings).generate("Reply with OK only.", "OK")
    except LLMConfigurationError:
        print("Configured model reachable: UNKNOWN")
        print("Diagnostic result: missing configuration")
        return 1
    except LLMAuthenticationError:
        print("Configured model reachable: NO")
        print("Diagnostic result: authentication failure")
        return 1
    except LLMQuotaError:
        print("Configured model reachable: UNKNOWN")
        print("Diagnostic result: quota or rate limit failure")
        return 1
    except LLMModelUnavailableError:
        print("Configured model reachable: NO")
        print("Diagnostic result: model unavailable")
        return 1
    except LLMRequestError:
        print("Configured model reachable: UNKNOWN")
        print("Diagnostic result: provider request failure")
        return 1

    print("Configured model reachable: YES")
    print("Diagnostic result: generation succeeded")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())