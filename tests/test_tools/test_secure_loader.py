"""Tests for src.tools.common.secure_loader - SecureDataLoader."""

from __future__ import annotations

import json
import os

import pytest

from src.tools.common.secure_loader import SecureDataLoader


class TestPathTraversal:
    """Tests for path traversal prevention."""

    def test_path_traversal_blocked(self, tmp_data_dir):
        """Paths with '..' that escape base_dir are rejected."""
        loader = SecureDataLoader(base_dir=str(tmp_data_dir))
        with pytest.raises(ValueError, match="escapes base directory"):
            loader.validate_path(str(tmp_data_dir / ".." / "etc" / "passwd"))

    def test_absolute_path_outside_base(self, tmp_data_dir):
        """Absolute paths outside base_dir are rejected."""
        loader = SecureDataLoader(base_dir=str(tmp_data_dir))
        with pytest.raises(ValueError, match="escapes base directory"):
            loader.validate_path("/etc/passwd")

    def test_valid_path_within_base(self, tmp_data_dir):
        """Valid paths within base_dir are accepted."""
        loader = SecureDataLoader(base_dir=str(tmp_data_dir))
        csv_path = str(tmp_data_dir / "test_data.csv")
        resolved = loader.validate_path(csv_path)
        assert resolved.name == "test_data.csv"


class TestFileExtension:
    """Tests for file extension whitelist."""

    def test_invalid_extension_blocked(self, tmp_path):
        """Files with unsupported extensions are rejected."""
        loader = SecureDataLoader()
        bad_file = tmp_path / "malicious.exe"
        bad_file.write_text("data")
        with pytest.raises(ValueError, match="Unsupported file type"):
            loader.validate_path(str(bad_file))

    def test_csv_allowed(self, tmp_path):
        f = tmp_path / "data.csv"
        f.write_text("a,b\n1,2")
        loader = SecureDataLoader()
        p = loader.validate_path(str(f))
        assert p.suffix == ".csv"

    def test_json_allowed(self, tmp_path):
        f = tmp_path / "data.json"
        f.write_text("{}")
        loader = SecureDataLoader()
        p = loader.validate_path(str(f))
        assert p.suffix == ".json"

    def test_parquet_allowed(self, tmp_path):
        f = tmp_path / "data.parquet"
        f.write_bytes(b"")
        loader = SecureDataLoader()
        p = loader.validate_path(str(f))
        assert p.suffix == ".parquet"

    def test_txt_blocked(self, tmp_path):
        f = tmp_path / "data.txt"
        f.write_text("text")
        loader = SecureDataLoader()
        with pytest.raises(ValueError, match="Unsupported file type"):
            loader.validate_path(str(f))


class TestFileSizeLimit:
    """Tests for file size validation."""

    def test_file_size_limit(self, tmp_path):
        """Files exceeding the size limit are rejected."""
        big_file = tmp_path / "big.csv"
        # Write a file larger than 100 bytes
        big_file.write_text("x" * 200)
        loader = SecureDataLoader(max_file_size=100)
        with pytest.raises(ValueError, match="too large"):
            loader.validate_path(str(big_file))

    def test_small_file_passes(self, tmp_path):
        small_file = tmp_path / "small.csv"
        small_file.write_text("a,b\n1,2")
        loader = SecureDataLoader(max_file_size=1024 * 1024)
        p = loader.validate_path(str(small_file))
        assert p.name == "small.csv"


class TestSecureLoad:
    """Tests for the full load pipeline."""

    def test_load_csv(self, tmp_data_dir):
        loader = SecureDataLoader(base_dir=str(tmp_data_dir))
        result = loader.load(str(tmp_data_dir / "test_data.csv"))
        assert "user_logs" in result
        assert len(result["user_logs"]) == 2  # 2 data rows

    def test_load_json(self, tmp_data_dir):
        loader = SecureDataLoader(base_dir=str(tmp_data_dir))
        result = loader.load(str(tmp_data_dir / "test_data.json"))
        assert "user_logs" in result

    def test_load_nonexistent_raises(self, tmp_path):
        loader = SecureDataLoader(base_dir=str(tmp_path))
        with pytest.raises(FileNotFoundError):
            loader.load(str(tmp_path / "missing.csv"))

    def test_load_json_content_valid(self, tmp_data_dir):
        loader = SecureDataLoader(base_dir=str(tmp_data_dir))
        content = loader.load_json(str(tmp_data_dir / "test_data.json"))
        assert content["key"] == "value"
        assert content["count"] == 42

    def test_validate_json_content_invalid(self, tmp_path):
        bad_json = tmp_path / "bad.json"
        bad_json.write_text("not json at all {")
        loader = SecureDataLoader(base_dir=str(tmp_path))
        with pytest.raises(ValueError, match="Invalid JSON"):
            loader.load_json(str(bad_json))

    def test_safe_list_files(self, tmp_data_dir):
        loader = SecureDataLoader(base_dir=str(tmp_data_dir))
        files = loader.safe_list_files()
        names = [f.name for f in files]
        assert "test_data.csv" in names
        assert "test_data.json" in names

    def test_safe_list_files_no_base_dir(self):
        loader = SecureDataLoader()
        with pytest.raises(ValueError, match="base_dir must be set"):
            loader.safe_list_files()
