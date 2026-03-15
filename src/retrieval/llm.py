"""
@author: Alok Katiyar
Seattle University, ARIN 5360
@see: https://catalog.seattleu.edu/preview_course_nopop.php?catoid=55&coid
=190380
@version: 2.0.0+w26
"""

# src/retrieval/llm.py

from typing import Optional

import httpx


class LLMClientError(Exception):
    """Raised when an LLM request fails."""


class LLMClient:
    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model: str = "qwen2.5:3b",
        api_key: Optional[str] = None,
        timeout: float = 30.0,
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.timeout = timeout

    def _headers(self) -> dict:
        """
        Always send Authorization header.
        Ollama does not validate it, so we use 'ollama' by default.
        (This also keeps the client compatible with OpenAI-style endpoints.)
        """
        key = self.api_key or "ollama"
        return {
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
        }

    def is_available(self) -> bool:
        """
        Check whether the LLM service is reachable.
        """
        try:
            with httpx.Client(timeout=self.timeout) as client:
                response = client.get(
                    f"{self.base_url}/v1/models",
                    headers=self._headers(),
                )
            return 200 <= response.status_code < 300
        except (httpx.RequestError, httpx.TimeoutException):
            return False

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 500,
    ) -> str:
        if not prompt or not isinstance(prompt, str):
            raise ValueError("Prompt must be a non-empty string")

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        try:
            with httpx.Client(timeout=self.timeout) as client:
                response = client.post(
                    f"{self.base_url}/v1/chat/completions",
                    json=payload,
                    headers=self._headers(),
                )
        except httpx.TimeoutException as e:
            raise LLMClientError("LLM request timed out") from e
        except httpx.RequestError as e:
            raise LLMClientError("Failed to reach LLM service") from e

        if response.status_code < 200 or response.status_code >= 300:
            raise LLMClientError(f"LLM error {response.status_code}: {response.text}")

        try:
            data = response.json()
            return data["choices"][0]["message"]["content"].strip()
        except Exception as e:
            raise LLMClientError(f"Unexpected LLM response format: {response.text}") from e
