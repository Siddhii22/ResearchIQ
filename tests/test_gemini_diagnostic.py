"""Ensure the optional Gemini diagnostic prints no secret or provider output."""

from types import SimpleNamespace

from scripts import check_gemini


def test_gemini_diagnostic_never_prints_key_or_response(monkeypatch, capsys) -> None:
    secret = "do-not-print-this-key"
    provider_text = "do-not-print-provider-response"

    class FakeGemini:
        def __init__(self, configuration: object) -> None:
            pass

        def generate(self, system_prompt: str, user_prompt: str) -> str:
            return provider_text

    monkeypatch.setattr(
        check_gemini,
        "settings",
        SimpleNamespace(gemini_api_key=secret, gemini_model="mock-gemini-model"),
    )
    monkeypatch.setattr(check_gemini, "GeminiLLM", FakeGemini)

    assert check_gemini.main() == 0
    output = capsys.readouterr().out

    assert "GEMINI_API_KEY present: YES" in output
    assert "Configured model reachable: YES" in output
    assert secret not in output
    assert provider_text not in output