import re
from dataclasses import dataclass


@dataclass
class GuardrailResult:
    passed: bool
    reason: str = ""
    sanitized_input: str = ""


INJECTION_PATTERNS = [
    # English: explicit instruction-override phrases only — bare words like
    # "pretend" or "act as" false-positive on legitimate business queries.
    r"(?i)(ignore\s+(all\s+)?(previous|prior|above)\s+(instructions?|prompts?|rules?))",
    r"(?i)(disregard\s+(your|all|the)\s+(system\s+)?(instructions?|prompts?|rules?))",
    r"(?i)(forget\s+(everything|all)\s+(you|above))",
    r"(?i)(you\s+are\s+now\s+(a|an)\s+(different|new|unrestricted))",
    r"(?i)(jailbreak|DAN\s+mode|developer\s+mode\s+enabled)",
    r"(?i)(reveal|show|print|repeat|output).{0,30}(system\s+prompt|secret\s+prompt)",
    r"(?i)(<\|?(im_start|system)\|?>|</?system>|```system)",
    # Chinese equivalents
    r"忽略.{0,6}(指令|提示词?|规则|设定)",
    r"(忽略|无视)(你的)?系统提示",
    r"(扮演|假装成)(系统|管理员|开发者)",
    r"(越狱|解锁开发者模式|显示你的系统提示)",
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


def validate_input(
    query: str,
    budget: float | None = None,
    scope: str | None = None,
) -> GuardrailResult:
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

    # Scope validation
    VALID_SCOPES = {"prospect", "conversion", "subsidy", "retention", "ad", "full", ""}
    if scope is not None and scope not in VALID_SCOPES:
        return GuardrailResult(
            passed=False,
            reason=f"Invalid scope: '{scope}'. Must be one of: {', '.join(sorted(VALID_SCOPES))}",
        )

    # Minimum budget check for subsidy scope
    if scope == "subsidy" and (budget is None or budget <= 0):
        return GuardrailResult(
            passed=False,
            reason="Budget must be greater than 0 for subsidy scope",
        )

    return GuardrailResult(passed=True, sanitized_input=query.strip())
