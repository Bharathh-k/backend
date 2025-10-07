"""Application configuration helpers."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Dict


_PROJECT_ROOT = Path(__file__).resolve().parent
_ENV_FILENAMES = (".env",)


class Settings:
    """Central application settings with environment overrides."""

    def __init__(self) -> None:
        self._env_from_file = self._load_env_file()

    @property
    def project_root(self) -> Path:
        return _PROJECT_ROOT

    @property
    def data_dir(self) -> Path:
        return self.project_root / "data"

    @property
    def prompts_dir(self) -> Path:
        return self.project_root / "prompts"

    def _load_env_file(self) -> Dict[str, str]:
        """Load key/value pairs from a local .env file for development."""
        locations = [
            self.project_root / filename
            for filename in _ENV_FILENAMES
        ] + [
            self.project_root.parent / filename
            for filename in _ENV_FILENAMES
        ] + [
            self.project_root.parent.parent / filename
            for filename in _ENV_FILENAMES
        ]

        env_data: Dict[str, str] = {}
        for candidate in locations:
            if not candidate.exists() or not candidate.is_file():
                continue
            for raw_line in candidate.read_text(encoding="utf-8").splitlines():
                line = raw_line.strip()
                if not line or line.startswith("#"):
                    continue
                if "=" not in line:
                    continue
                key, value = line.split("=", 1)
                key = key.strip().upper()
                if not key:
                    continue
                env_data[key] = value.strip()
        return env_data

    def _get(self, key: str) -> str:
        env_key = key.upper()
        value = os.getenv(env_key)
        if value is None or not value.strip():
            value = self._env_from_file.get(env_key)
        if value is None or not value.strip():
            raise RuntimeError(
                f"Required environment variable '{env_key}' is not configured."
            )
        return value.strip()

    @property
    def mongo_uri(self) -> str:
        return self._get("mongo_uri")

    @property
    def gemini_api_key(self) -> str:
        return self._get("gemini_api_key")

    @property
    def reddit_client_id(self) -> str:
        return self._get("reddit_client_id")

    @property
    def reddit_client_secret(self) -> str:
        return self._get("reddit_client_secret")

    @property
    def reddit_user_agent(self) -> str:
        return self._get("reddit_user_agent")

    def ensure_directories(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.prompts_dir.mkdir(parents=True, exist_ok=True)

    def get_data_path(self, filename: str) -> Path:
        return self.data_dir / filename

    def get_prompt_path(self, filename: str) -> Path:
        return self.prompts_dir / filename


settings = Settings()

SettingsType = Settings
