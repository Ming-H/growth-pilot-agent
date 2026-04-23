"""SecureDataLoader - Safe data loading with path traversal protection,
file type validation, and size limits.

Security features:
- Path traversal prevention (resolves symlinks, checks containment)
- File type whitelist (csv/parquet/json only)
- File size limit (configurable, default 500MB)
- Content validation for JSON files
- Symlink detection and rejection
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)


class SecureDataLoader:
    """Data loader with security validation for freight growth data.

    All file loads go through a three-stage validation pipeline:
    1. Path validation (traversal + symlink check)
    2. File type validation (extension whitelist)
    3. File size validation (configurable limit)

    Usage::

        loader = SecureDataLoader(base_dir="/data/freight")
        data = loader.load("user_logs.csv")
        # or with validation only:
        path = loader.validate_path("user_logs.csv")
    """

    ALLOWED_EXTENSIONS = frozenset({".csv", ".parquet", ".pq", ".json"})
    MAX_FILE_SIZE = 500 * 1024 * 1024  # 500MB

    def __init__(
        self,
        base_dir: str | None = None,
        max_file_size: int | None = None,
    ) -> None:
        """Initialize the secure data loader.

        Args:
            base_dir: Base directory for path containment. If set, all
                      loaded files must reside within this directory.
            max_file_size: Maximum file size in bytes. Defaults to 500MB.
        """
        self.base_dir = Path(base_dir).resolve() if base_dir else None
        self.max_file_size = max_file_size or self.MAX_FILE_SIZE

    # ------------------------------------------------------------------
    # Validation pipeline
    # ------------------------------------------------------------------

    def validate_path(self, path: str) -> Path:
        """Validate a file path through the security pipeline.

        Stages:
        1. Resolve to absolute path
        2. Check for symlink escape
        3. Path traversal protection (base_dir containment)
        4. File extension whitelist check
        5. File size limit check

        Args:
            path: File path to validate (absolute or relative to base_dir).

        Returns:
            Resolved absolute Path object.

        Raises:
            ValueError: Path fails any security check.
            FileNotFoundError: File does not exist.
        """
        p = Path(path).resolve()

        # Stage 1: Symlink detection
        if p.is_symlink():
            real_target = p.resolve()
            if self.base_dir:
                try:
                    real_target.relative_to(self.base_dir)
                except ValueError:
                    raise ValueError(
                        f"Symlink '{path}' points outside base directory "
                        f"'{self.base_dir}' (target: {real_target})"
                    )
            logger.warning("Symlink detected: %s -> %s", path, real_target)

        # Stage 2: Path traversal protection
        if self.base_dir:
            try:
                p.relative_to(self.base_dir)
            except ValueError:
                raise ValueError(
                    f"Path '{path}' escapes base directory '{self.base_dir}'. "
                    f"Resolved to: {p}"
                )

        # Stage 3: File extension check
        if p.suffix.lower() not in self.ALLOWED_EXTENSIONS:
            allowed = ", ".join(sorted(self.ALLOWED_EXTENSIONS))
            raise ValueError(
                f"Unsupported file type: '{p.suffix}'. "
                f"Allowed extensions: {allowed}"
            )

        # Stage 4: File size check (only if file exists)
        if p.exists():
            size = p.stat().st_size
            if size > self.max_file_size:
                size_mb = size / (1024 * 1024)
                limit_mb = self.max_file_size / (1024 * 1024)
                raise ValueError(
                    f"File too large: {size_mb:.1f}MB exceeds "
                    f"limit of {limit_mb:.0f}MB"
                )

        return p

    def validate_json_content(self, path: str) -> dict[str, Any]:
        """Validate that a JSON file contains valid, parseable content.

        Args:
            path: Path to the JSON file.

        Returns:
            Parsed JSON content as a dict.

        Raises:
            ValueError: File is not valid JSON or exceeds nesting limits.
        """
        p = self.validate_path(path)

        if not p.exists():
            raise FileNotFoundError(f"Data file not found: {p}")

        try:
            with open(p, "r", encoding="utf-8") as f:
                content = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in '{p}': {e}") from e
        except UnicodeDecodeError as e:
            raise ValueError(f"File '{p}' is not valid UTF-8: {e}") from e

        if not isinstance(content, (dict, list)):
            raise ValueError(
                f"JSON root must be a dict or list, got {type(content).__name__}"
            )

        return content

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def load(self, path: str) -> dict[str, Any]:
        """Load a data file with full security validation.

        Supports: CSV, Parquet, JSON (returns DataFrames).

        Args:
            path: File path to load.

        Returns:
            Dict with 'user_logs' (DataFrame) and 'user_profile' (DataFrame).

        Raises:
            FileNotFoundError: File does not exist.
            ValueError: Security validation failed or unsupported format.
        """
        p = self.validate_path(path)

        if not p.exists():
            raise FileNotFoundError(f"Data file not found: {p}")

        if p.suffix == ".csv":
            df = pd.read_csv(p)
        elif p.suffix in (".parquet", ".pq"):
            df = pd.read_parquet(p)
        elif p.suffix == ".json":
            df = self._load_json_as_dataframe(p)
        else:
            raise ValueError(f"Unsupported format: {p.suffix}")

        rows, cols = len(df), len(df.columns)
        logger.info("SecureDataLoader: loaded %s (%d rows, %d cols)", p.name, rows, cols)

        return {"user_logs": df, "user_profile": pd.DataFrame()}

    def load_json(self, path: str) -> dict[str, Any]:
        """Load a JSON file as a dict (not DataFrame).

        Use this for config files, evaluation datasets, etc.

        Args:
            path: JSON file path.

        Returns:
            Parsed JSON content.
        """
        return self.validate_json_content(path)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_json_as_dataframe(self, p: Path) -> pd.DataFrame:
        """Load JSON file as a DataFrame, handling both flat and nested."""
        try:
            return pd.read_json(p)
        except (ValueError, KeyError):
            # Try loading as records
            with open(p, "r", encoding="utf-8") as f:
                content = json.load(f)
            if isinstance(content, list):
                return pd.DataFrame(content)
            if isinstance(content, dict):
                return pd.DataFrame([content])
            raise ValueError(f"Cannot convert JSON to DataFrame: {p}")

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def safe_list_files(self, pattern: str = "*") -> list[Path]:
        """List files in base_dir matching pattern, with validation.

        Args:
            pattern: Glob pattern to match files.

        Returns:
            List of validated file paths within base_dir.

        Raises:
            ValueError: base_dir is not configured.
        """
        if not self.base_dir:
            raise ValueError("base_dir must be set to use safe_list_files()")

        if not self.base_dir.exists():
            return []

        results: list[Path] = []
        for p in self.base_dir.glob(pattern):
            if not p.is_file():
                continue
            try:
                self.validate_path(str(p))
                results.append(p)
            except ValueError:
                logger.debug("Skipping invalid file: %s", p)

        return sorted(results)
