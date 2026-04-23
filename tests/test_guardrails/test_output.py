"""Tests for src.guardrails.output_guard - output validation and sensitive info detection."""
from __future__ import annotations

import pytest

from src.guardrails.output_guard import (
    OutputGuardResult,
    check_sensitive_info,
    validate_output,
)


# ---------------------------------------------------------------------------
# validate_output - valid outputs
# ---------------------------------------------------------------------------


class TestValidateOutputValid:
    """Tests for valid outputs that should pass validation."""

    def test_valid_output_passes(self):
        """A normal, substantive output passes validation."""
        output = "# Analysis Report\n\nThe user acquisition rate improved by 15% this quarter."
        result = validate_output(output)
        assert result.passed is True
        assert result.sanitized_output == output

    def test_clean_output_passes_through(self):
        """Clean output is returned as sanitized_output."""
        output = "Growth analysis complete. Key metrics: CAC $45, LTV $320, retention 85%."
        result = validate_output(output)
        assert result.passed is True
        assert result.sanitized_output == output.strip()


# ---------------------------------------------------------------------------
# validate_output - short outputs
# ---------------------------------------------------------------------------


class TestValidateOutputShort:
    """Tests for short/empty output rejection."""

    def test_short_output_rejected(self):
        """Output shorter than 10 characters is rejected."""
        result = validate_output("short")
        assert result.passed is False
        assert "short" in result.reason.lower()

    def test_empty_output_rejected(self):
        """Empty string output is rejected."""
        result = validate_output("")
        assert result.passed is False

    def test_none_output_rejected(self):
        """None output is rejected."""
        result = validate_output(None)  # type: ignore
        assert result.passed is False

    def test_whitespace_only_output_rejected(self):
        """Whitespace-only output is rejected."""
        result = validate_output("          ")
        assert result.passed is False

    def test_output_at_boundary(self):
        """Output with exactly 10 non-whitespace characters passes."""
        result = validate_output("1234567890")
        assert result.passed is True

    def test_output_below_boundary(self):
        """Output with 9 non-whitespace characters is rejected."""
        result = validate_output("123456789")
        assert result.passed is False


# ---------------------------------------------------------------------------
# check_sensitive_info - sensitive pattern detection
# ---------------------------------------------------------------------------


class TestCheckSensitiveInfo:
    """Tests for sensitive information detection in output."""

    def test_password_detected(self):
        """Outputs containing password assignments are flagged."""
        result = check_sensitive_info("The database password=secret123 needs rotation")
        assert result.passed is False
        assert "sensitive" in result.reason.lower()

    def test_api_key_detected(self):
        """Outputs containing api_key assignments are flagged."""
        result = check_sensitive_info("Use api_key=sk-abc123def456 for the API")
        assert result.passed is False

    def test_api_key_hyphen_detected(self):
        """api-key with hyphen is detected."""
        result = check_sensitive_info("Set api-key: sk-abc123 for authentication")
        assert result.passed is False

    def test_secret_detected(self):
        """Outputs containing secret assignments are flagged."""
        result = check_sensitive_info("The secret=mysecretvalue is exposed")
        assert result.passed is False

    def test_token_detected(self):
        """Outputs containing token assignments are flagged."""
        result = check_sensitive_info("Bearer token=eyJhbGciOiJIUzI1NiJ9.xxx")
        assert result.passed is False

    def test_credit_card_detected(self):
        """Outputs containing 16-digit numbers are flagged as credit cards."""
        result = check_sensitive_info("Card number: 4111111111111111 on file")
        assert result.passed is False

    def test_clean_output_passes(self):
        """Normal analysis output without sensitive data passes."""
        output = "The conversion rate improved from 2.1% to 3.5% after implementing the new strategy."
        result = check_sensitive_info(output)
        assert result.passed is True

    def test_case_insensitive_password(self):
        """Password detection is case-insensitive."""
        result = check_sensitive_info("PASSWORD=admin123")
        assert result.passed is False

    def test_case_insensitive_api_key(self):
        """API_KEY detection is case-insensitive."""
        result = check_sensitive_info("API_KEY=sk-test-key")
        assert result.passed is False

    def test_safe_numbers_pass(self):
        """Regular numbers (not 16 digits) pass through."""
        result = check_sensitive_info("Revenue was 1250000 and users 4500")
        assert result.passed is True

    def test_short_digit_sequences_pass(self):
        """Numbers with fewer than 16 digits are not flagged."""
        result = check_sensitive_info("Order ID: 123456789012345")
        assert result.passed is True
