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


def check_sensitive_info(text: str) -> OutputGuardResult:
    """Check text for potential sensitive information leaks."""
    for pattern in SENSITIVE_PATTERNS:
        if re.search(pattern, text):
            return OutputGuardResult(
                passed=False,
                reason="Sensitive information detected in output",
            )
    return OutputGuardResult(passed=True)


def validate_output(output: str) -> OutputGuardResult:
    """Validate LLM output before returning to user."""
    if not output or len(output.strip()) < 10:
        return OutputGuardResult(passed=False, reason="Output too short")
    sensitive_check = check_sensitive_info(output)
    if not sensitive_check.passed:
        return sensitive_check
    return OutputGuardResult(passed=True, sanitized_output=output.strip())
