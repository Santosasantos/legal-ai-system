"""
LLM Provider abstraction.
Switch provider via LLM_PROVIDER env var: ollama | openai | gemini | anthropic
All providers expose the same interface: generate() and embed().
"""

from abc import ABC, abstractmethod
from typing import List, Optional
import httpx
import json
import base64
from pathlib import Path

from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)


# ── Base Interface ────────────────────────────────────────────────────────────

class BaseLLMProvider(ABC):
    @abstractmethod
    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_tokens: int = 2048,
        temperature: float = 0.1,
    ) -> str:
        """Generate text from a prompt."""
        ...

    @abstractmethod
    async def generate_with_image(
        self,
        prompt: str,
        image_path: str,
        system_prompt: Optional[str] = None,
        max_tokens: int = 2048,
    ) -> str:
        """Vision: generate text from prompt + image (for OCR)."""
        ...

    @abstractmethod
    async def embed(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of texts, return list of float vectors."""
        ...

    @property
    @abstractmethod
    def provider_name(self) -> str:
        ...

    @property
    @abstractmethod
    def text_model(self) -> str:
        ...

    @property
    @abstractmethod
    def vision_model(self) -> str:
        ...

    @property
    @abstractmethod
    def embed_model(self) -> str:
        ...


# ── Ollama Provider (default — Qwen2.5-VL + Qwen2.5) ─────────────────────────

class OllamaProvider(BaseLLMProvider):
    def __init__(self):
        self.base_url = settings.OLLAMA_BASE_URL
        self._text_model = settings.OLLAMA_TEXT_MODEL
        self._vision_model = settings.OLLAMA_VISION_MODEL
        self._embed_model = settings.OLLAMA_EMBED_MODEL

    @property
    def provider_name(self) -> str:
        return "ollama"

    @property
    def text_model(self) -> str:
        return self._text_model

    @property
    def vision_model(self) -> str:
        return self._vision_model

    @property
    def embed_model(self) -> str:
        return self._embed_model

    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_tokens: int = 2048,
        temperature: float = 0.1,
    ) -> str:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": self._text_model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }
        async with httpx.AsyncClient(timeout=300.0) as client:
            resp = await client.post(
                f"{self.base_url}/api/chat", json=payload
            )
            resp.raise_for_status()
            data = resp.json()
            return data["message"]["content"]

    async def generate_with_image(
        self,
        prompt: str,
        image_path: str,
        system_prompt: Optional[str] = None,
        max_tokens: int = 2048,
    ) -> str:
        """Use Qwen2.5-VL for vision-based OCR and document understanding."""
        with open(image_path, "rb") as f:
            image_b64 = base64.b64encode(f.read()).decode("utf-8")

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({
            "role": "user",
            "content": prompt,
            "images": [image_b64],
        })

        payload = {
            "model": self._vision_model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": 0.0,   # deterministic for OCR
                "num_predict": max_tokens,
            },
        }
        async with httpx.AsyncClient(timeout=600.0) as client:
            resp = await client.post(
                f"{self.base_url}/api/chat", json=payload
            )
            resp.raise_for_status()
            data = resp.json()
            return data["message"]["content"]

    async def embed(self, texts: List[str]) -> List[List[float]]:
        embeddings = []
        async with httpx.AsyncClient(timeout=120.0) as client:
            for text in texts:
                payload = {"model": self._embed_model, "prompt": text}
                resp = await client.post(
                    f"{self.base_url}/api/embeddings", json=payload
                )
                resp.raise_for_status()
                embeddings.append(resp.json()["embedding"])
        return embeddings


# ── OpenAI Provider ───────────────────────────────────────────────────────────

class OpenAIProvider(BaseLLMProvider):
    def __init__(self):
        try:
            from openai import AsyncOpenAI
            self.client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
        except ImportError:
            raise ImportError("Install openai: pip install openai")
        self._text_model = settings.OPENAI_TEXT_MODEL
        self._embed_model = settings.OPENAI_EMBED_MODEL

    @property
    def provider_name(self) -> str:
        return "openai"

    @property
    def text_model(self) -> str:
        return self._text_model

    @property
    def vision_model(self) -> str:
        return self._text_model  # GPT-4o handles vision too

    @property
    def embed_model(self) -> str:
        return self._embed_model

    async def generate(self, prompt, system_prompt=None, max_tokens=2048, temperature=0.1):
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        resp = await self.client.chat.completions.create(
            model=self._text_model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return resp.choices[0].message.content

    async def generate_with_image(self, prompt, image_path, system_prompt=None, max_tokens=2048):
        with open(image_path, "rb") as f:
            image_b64 = base64.b64encode(f.read()).decode("utf-8")
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_b64}"}},
            ],
        })
        resp = await self.client.chat.completions.create(
            model=self._text_model,
            messages=messages,
            max_tokens=max_tokens,
        )
        return resp.choices[0].message.content

    async def embed(self, texts: List[str]) -> List[List[float]]:
        resp = await self.client.embeddings.create(
            model=self._embed_model, input=texts
        )
        return [item.embedding for item in resp.data]


# ── Gemini Provider ───────────────────────────────────────────────────────────

class GeminiProvider(BaseLLMProvider):
    """
    Google Gemini provider — default for fast startup (no model downloads).
    Uses gemini-1.5-flash for both text and vision (OCR), text-embedding-004
    for embeddings.

    Free tier: 15 RPM / 1M tokens per day — sufficient for evaluation.
    Get a free API key at: https://aistudio.google.com/app/apikey
    """

    def __init__(self):
        if not settings.GEMINI_API_KEY:
            raise ValueError(
                "GEMINI_API_KEY is not set.\n"
                "Get a free key at https://aistudio.google.com/app/apikey\n"
                "Then set it in your .env file: GEMINI_API_KEY=your-key-here"
            )
        try:
            import google.generativeai as genai
            genai.configure(api_key=settings.GEMINI_API_KEY)
            self.genai = genai
            self._model_name = settings.GEMINI_TEXT_MODEL
            self._embed_model_name = "models/gemini-embedding-001"
            # GenerativeModel is lightweight to construct
            self._text_client = genai.GenerativeModel(
                model_name=self._model_name,
                generation_config={
                    "temperature": 0.1,
                    "max_output_tokens": 4096,
                },
                safety_settings=[
                    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
                    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
                    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
                    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
                ],
            )
        except ImportError:
            raise ImportError(
                "google-generativeai is not installed. "
                "Run: pip install google-generativeai==0.8.3"
            )

    @property
    def provider_name(self) -> str:
        return "gemini"

    @property
    def text_model(self) -> str:
        return self._model_name

    @property
    def vision_model(self) -> str:
        return self._model_name  # gemini-1.5-flash handles vision natively

    @property
    def embed_model(self) -> str:
        return self._embed_model_name

    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_tokens: int = 2048,
        temperature: float = 0.1,
    ) -> str:
        import asyncio
        full_prompt = f"{system_prompt}\n\n{prompt}" if system_prompt else prompt
        # google-generativeai SDK is sync; run in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        resp = await loop.run_in_executor(
            None,
            lambda: self._text_client.generate_content(
                full_prompt,
                generation_config={"temperature": temperature, "max_output_tokens": max_tokens},
            ),
        )
        return resp.text

    async def generate_with_image(
        self,
        prompt: str,
        image_path: str,
        system_prompt: Optional[str] = None,
        max_tokens: int = 2048,
    ) -> str:
        """Use Gemini 1.5 Flash vision for OCR and document understanding."""
        import asyncio
        import PIL.Image

        img = PIL.Image.open(image_path)
        full_prompt = f"{system_prompt}\n\n{prompt}" if system_prompt else prompt

        loop = asyncio.get_event_loop()
        resp = await loop.run_in_executor(
            None,
            lambda: self._text_client.generate_content(
                [full_prompt, img],
                generation_config={"temperature": 0.0, "max_output_tokens": max_tokens},
            ),
        )
        return resp.text

    async def embed(self, texts: List[str]) -> List[List[float]]:
        """
        Embed texts using text-embedding-004.
        Uses retrieval_document task type for indexing, retrieval_query for queries.
        Batches up to 100 texts per call to stay within API limits.
        """
        import asyncio

        results: List[List[float]] = []
        batch_size = 20  # conservative batch size for free tier

        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            loop = asyncio.get_event_loop()

            batch_results = await loop.run_in_executor(
                None,
                lambda b=batch: [
                    self.genai.embed_content(
                        model=self._embed_model_name,
                        content=text,
                        task_type="retrieval_document",
                    )["embedding"]
                    for text in b
                ],
            )
            results.extend(batch_results)

        return results


# ── Anthropic Provider ────────────────────────────────────────────────────────

class AnthropicProvider(BaseLLMProvider):
    def __init__(self):
        try:
            import anthropic
            self.client = anthropic.AsyncAnthropic(api_key=settings.ANTHROPIC_API_KEY)
        except ImportError:
            raise ImportError("Install anthropic: pip install anthropic")

    @property
    def provider_name(self) -> str:
        return "anthropic"

    @property
    def text_model(self) -> str:
        return settings.ANTHROPIC_TEXT_MODEL

    @property
    def vision_model(self) -> str:
        return settings.ANTHROPIC_TEXT_MODEL

    @property
    def embed_model(self) -> str:
        return "voyage-3"  # Anthropic uses Voyage for embeddings

    async def generate(self, prompt, system_prompt=None, max_tokens=2048, temperature=0.1):
        kwargs = {"model": self.text_model, "max_tokens": max_tokens,
                  "messages": [{"role": "user", "content": prompt}]}
        if system_prompt:
            kwargs["system"] = system_prompt
        resp = await self.client.messages.create(**kwargs)
        return resp.content[0].text

    async def generate_with_image(self, prompt, image_path, system_prompt=None, max_tokens=2048):
        with open(image_path, "rb") as f:
            image_b64 = base64.b64encode(f.read()).decode("utf-8")
        content = [
            {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": image_b64}},
            {"type": "text", "text": prompt},
        ]
        kwargs = {"model": self.text_model, "max_tokens": max_tokens,
                  "messages": [{"role": "user", "content": content}]}
        if system_prompt:
            kwargs["system"] = system_prompt
        resp = await self.client.messages.create(**kwargs)
        return resp.content[0].text

    async def embed(self, texts: List[str]) -> List[List[float]]:
        raise NotImplementedError(
            "Anthropic embedding requires Voyage API. Set LLM_PROVIDER=ollama for local embeddings."
        )


# ── Factory ───────────────────────────────────────────────────────────────────

def get_llm_provider() -> BaseLLMProvider:
    """Return the configured LLM provider instance."""
    provider = settings.LLM_PROVIDER
    logger.info(f"Initialising LLM provider: {provider}")
    if provider == "ollama":
        return OllamaProvider()
    elif provider == "openai":
        return OpenAIProvider()
    elif provider == "gemini":
        return GeminiProvider()
    elif provider == "anthropic":
        return AnthropicProvider()
    else:
        raise ValueError(f"Unknown LLM_PROVIDER: {provider}. Choose from: ollama, openai, gemini, anthropic")


# Singleton — reused across requests
_provider_instance: Optional[BaseLLMProvider] = None


def get_provider() -> BaseLLMProvider:
    global _provider_instance
    if _provider_instance is None:
        _provider_instance = get_llm_provider()
    return _provider_instance
