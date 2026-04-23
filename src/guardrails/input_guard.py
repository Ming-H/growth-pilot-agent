import re
from dataclasses import dataclass


@dataclass
class GuardrailResult:
    passed: bool
    reason: str = ""
    sanitized_input: str = ""


INJECTION_PATTERNS = [
    r"(?i)(ignore\s+previous|forget\s+your|you\s+are\s+now|system\s*prompt)",
    r"(?i)(pretend|act\s+as|roleplay|jailbreak)",
    r"(?i)(<system>|</system>|```system)",
    r"(?i)(translate.*above|summarize.*above|repeat.*above)",
]


def check_prompt_injection(text: str) -> GuardrailResult:
    """Check text for potential prompt injection patterns."""
    for pattern in INJECTION_PATTERNS:
        if re.search(pattern, text):
            return GuardrailResult(
                passed=False,
                reason="Potential prompt injection detected",
            )
    return GuardrailResult(passed=True, sanitized_input=text)


def validate_input(query: str, budget: float | None = None) -> GuardrailResult:
    """Validate user input before processing."""
    if not query or len(query.strip()) < 5:
        return GuardrailResult(passed=False, reason="Query too short")
    if len(query) > 5000:
        return GuardrailResult(passed=False, reason="Query too long (max 5000 chars)")

    injection_check = check_prompt_injection(query)
    if not injection_check.passed:
        return injection_check

    if budget is not None and (budget < 0 or budget > 1_000_000):
        return GuardrailResult(passed=False, reason="Budget out of valid range")

    return GuardrailResult(passed=True, sanitized_input=query.strip())
