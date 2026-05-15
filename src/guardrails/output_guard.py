import re
from dataclasses import dataclass


@dataclass
class OutputGuardResult:
    passed: bool
    reason: str = ""
    sanitized_output: str = ""


SENSITIVE_PATTERNS = [
    r"(?i)(password|api[_-]?key|secret|token)\s*[:=]\s*\S+",
    r"\b\d{16}\b",  # credit card
]

PII_PATTERNS = [
    r"\b\d{3}[-.\s]?\d{3,4}[-.\s]?\d{4}\b",  # phone numbers
    r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",  # email addresses
]

MAX_OUTPUT_LENGTH = 100_000


def check_sensitive_info(text: str) -> OutputGuardResult:
    """Check text for potential sensitive information leaks."""
    for pattern in SENSITIVE_PATTERNS:
        if re.search(pattern, text):
            return OutputGuardResult(
                passed=False,
                reason="Sensitive information detected in output",
            )
    return OutputGuardResult(passed=True)


def check_pii(text: str) -> OutputGuardResult:
    """Check text for personally identifiable information (PII)."""
    for pattern in PII_PATTERNS:
        if re.search(pattern, text):
            return OutputGuardResult(
                passed=False,
                reason="PII detected in output (phone number or email address)",
            )
    return OutputGuardResult(passed=True)


def validate_output(output: str) -> OutputGuardResult:
    """Validate LLM output before returning to user."""
    if not output or len(output.strip()) < 50:
        return OutputGuardResult(passed=False, reason="Output too short (minimum 50 chars)")

    if len(output) > MAX_OUTPUT_LENGTH:
        return OutputGuardResult(
            passed=False,
            reason=f"Output exceeds maximum length ({MAX_OUTPUT_LENGTH} chars)",
        )

    sensitive_check = check_sensitive_info(output)
    if not sensitive_check.passed:
        return sensitive_check

    pii_check = check_pii(output)
    if not pii_check.passed:
        return pii_check

    return OutputGuardResult(passed=True, sanitized_output=output.strip())
