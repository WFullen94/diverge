"""Thin model adapters — each returns a (str) -> str callable.

Simpler than llm-reliability's adapters because diverge only needs text output,
not confidence scores or logprobs.
"""

from __future__ import annotations

from abc import ABC, abstractmethod


class ModelAdapter(ABC):
    """Base class. Subclasses implement generate(); __call__ delegates to it."""

    @abstractmethod
    def generate(self, prompt: str) -> str:
        ...

    def __call__(self, prompt: str) -> str:
        return self.generate(prompt)

    @property
    def name(self) -> str:
        return self.__class__.__name__


class OpenAIAdapter(ModelAdapter):
    """OpenAI chat completions adapter (temperature=0 by default)."""

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        api_key: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 256,
        system_prompt: str = "You are a helpful assistant. Be concise.",
    ) -> None:
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.system_prompt = system_prompt
        self._client = None
        self._api_key = api_key

    def _get_client(self):
        if self._client is None:
            import openai
            self._client = openai.OpenAI(api_key=self._api_key)
        return self._client

    def generate(self, prompt: str) -> str:
        client = self._get_client()
        resp = client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": prompt},
            ],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        return resp.choices[0].message.content or ""

    @property
    def name(self) -> str:
        return self.model


class AnthropicAdapter(ModelAdapter):
    """Anthropic Messages API adapter."""

    def __init__(
        self,
        model: str = "claude-haiku-4-5-20251001",
        api_key: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 256,
    ) -> None:
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._client = None
        self._api_key = api_key

    def _get_client(self):
        if self._client is None:
            import anthropic
            self._client = anthropic.Anthropic(api_key=self._api_key)
        return self._client

    def generate(self, prompt: str) -> str:
        client = self._get_client()
        msg = client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            messages=[{"role": "user", "content": prompt}],
        )
        return msg.content[0].text if msg.content else ""

    @property
    def name(self) -> str:
        return self.model


class OllamaAdapter(ModelAdapter):
    """Ollama local inference adapter."""

    def __init__(
        self,
        model: str = "llama3",
        host: str = "http://localhost:11434",
        temperature: float = 0.0,
        max_tokens: int = 256,
    ) -> None:
        self.model = model
        self.host = host
        self.temperature = temperature
        self.max_tokens = max_tokens

    def generate(self, prompt: str) -> str:
        import ollama
        resp = ollama.generate(
            model=self.model,
            prompt=prompt,
            options={"temperature": self.temperature, "num_predict": self.max_tokens},
        )
        return resp.get("response", "")

    @property
    def name(self) -> str:
        return f"ollama/{self.model}"


class HuggingFaceAdapter(ModelAdapter):
    """HuggingFace transformers text-generation adapter (local)."""

    def __init__(
        self,
        model: str,
        device: str = "auto",
        temperature: float = 0.0,
        max_new_tokens: int = 256,
    ) -> None:
        self.model_id = model
        self.device = device
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        self._pipeline = None

    def _get_pipeline(self):
        if self._pipeline is None:
            from transformers import pipeline
            self._pipeline = pipeline(
                "text-generation",
                model=self.model_id,
                device_map=self.device,
            )
        return self._pipeline

    def generate(self, prompt: str) -> str:
        pipe = self._get_pipeline()
        do_sample = self.temperature > 0
        out = pipe(
            prompt,
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature if do_sample else None,
            do_sample=do_sample,
        )
        generated = out[0]["generated_text"]
        # Strip the prompt prefix that some pipelines echo back
        if generated.startswith(prompt):
            generated = generated[len(prompt):]
        return generated.strip()

    @property
    def name(self) -> str:
        return self.model_id
