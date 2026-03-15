"""
@author: Alok Katiyar
Seattle University, ARIN 5360
@see: https://catalog.seattleu.edu/preview_course_nopop.php?catoid=55&coid=190380
@version: 2.0.0+w26
"""

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
        timeout: float = 60.0,
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.timeout = timeout

    def _headers(self) -> dict:
        """
        Ollama does not require auth for local use, but keeping the header
        makes this client flexible for OpenAI-compatible gateways too.
        """
        key = self.api_key or "ollama"
        return {
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
        }

    def is_available(self) -> bool:
        """
        Check whether Ollama is reachable.
        Prefer the native Ollama tags endpoint because it is the most reliable.
        """
        try:
            with httpx.Client(timeout=10.0) as client:
                response = client.get(f"{self.base_url}/api/tags")
            return 200 <= response.status_code < 300
        except (httpx.RequestError, httpx.TimeoutException):
            return False

    def _generate_native_ollama(
        self,
        prompt: str,
        system_prompt: Optional[str],
        temperature: float,
        max_tokens: int,
    ) -> str:
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }

        if system_prompt and system_prompt.strip():
            payload["system"] = system_prompt.strip()

        try:
            with httpx.Client(timeout=self.timeout) as client:
                response = client.post(
                    f"{self.base_url}/api/generate",
                    json=payload,
                    headers=self._headers(),
                )
        except httpx.TimeoutException as e:
            raise LLMClientError("LLM request timed out") from e
        except httpx.RequestError as e:
            raise LLMClientError(f"Failed to reach LLM service: {e}") from e

        if not (200 <= response.status_code < 300):
            raise LLMClientError(f"Ollama error {response.status_code}: {response.text}")

        try:
            data = response.json()
        except Exception as e:
            raise LLMClientError(f"Invalid JSON from Ollama: {response.text}") from e

        text = data.get("response")
        if not isinstance(text, str) or not text.strip():
            raise LLMClientError(f"Unexpected Ollama response format: {data}")

        return text.strip()

    def _generate_openai_compatible(
        self,
        prompt: str,
        system_prompt: Optional[str],
        temperature: float,
        max_tokens: int,
    ) -> str:
        messages = []
        if system_prompt and system_prompt.strip():
            messages.append({"role": "system", "content": system_prompt.strip()})
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
            raise LLMClientError(f"Failed to reach LLM service: {e}") from e

        if not (200 <= response.status_code < 300):
            raise LLMClientError(
                f"OpenAI-compatible endpoint error {response.status_code}: {response.text}"
            )

        try:
            data = response.json()
            text = data["choices"][0]["message"]["content"]
        except Exception as e:
            raise LLMClientError(
                f"Unexpected OpenAI-compatible response format: {response.text}"
            ) from e

        if not isinstance(text, str) or not text.strip():
            raise LLMClientError(f"Empty response from model: {data}")

        return text.strip()

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 500,
    ) -> str:
        if not prompt or not isinstance(prompt, str):
            raise ValueError("Prompt must be a non-empty string")

        # First try native Ollama. If that fails because the server exposes
        # only the OpenAI-compatible surface, fall back automatically.
        native_error: Optional[str] = None

        try:
            return self._generate_native_ollama(
                prompt=prompt,
                system_prompt=system_prompt,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        except LLMClientError as e:
            native_error = str(e)

        try:
            return self._generate_openai_compatible(
                prompt=prompt,
                system_prompt=system_prompt,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        except LLMClientError as e:
            raise LLMClientError(
                "Both Ollama native and OpenAI-compatible generation failed. "
                f"Native error: {native_error} | Fallback error: {e}"
            ) from e
