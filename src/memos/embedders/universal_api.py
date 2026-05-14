import asyncio
import os
import time

from openai import AzureOpenAI as AzureClient
from openai import OpenAI as OpenAIClient

from memos.configs.embedder import UniversalAPIEmbedderConfig
from memos.embedders.base import BaseEmbedder
from memos.log import get_logger
from memos.utils import timed_with_status


logger = get_logger(__name__)


def _sanitize_unicode(text: str) -> str:
    """
    Remove Unicode surrogates and other problematic characters.
    Surrogates (U+D800-U+DFFF) cause UnicodeEncodeError with some APIs.
    """
    try:
        # Encode with 'surrogatepass' then decode, replacing invalid chars
        cleaned = text.encode("utf-8", errors="surrogatepass").decode("utf-8", errors="replace")
        # Replace replacement char with empty string for cleaner output
        return cleaned.replace("\ufffd", "")
    except Exception:
        # Fallback: remove all non-BMP characters
        return "".join(c for c in text if ord(c) < 0x10000)


class UniversalAPIEmbedder(BaseEmbedder):
    def __init__(self, config: UniversalAPIEmbedderConfig):
        self.provider = config.provider
        self.config = config

        if self.provider == "openai":
            self.client = OpenAIClient(
                api_key=config.api_key,
                base_url=config.base_url,
                default_headers=config.headers_extra if config.headers_extra else None,
            )
        elif self.provider == "azure":
            self.client = AzureClient(
                azure_endpoint=config.base_url,
                api_version="2024-03-01-preview",
                api_key=config.api_key,
            )
        else:
            raise ValueError(f"Embeddings unsupported provider: {self.provider}")
        self.use_backup_client = config.backup_client
        if self.use_backup_client:
            self.backup_client = OpenAIClient(
                api_key=config.backup_api_key,
                base_url=config.backup_base_url,
                default_headers=config.backup_headers_extra
                if config.backup_headers_extra
                else None,
            )

    def _get_model(self) -> str:
        """Get model name from config, falling back to default."""
        return getattr(self.config, "model_name_or_path", None) or "text-embedding-3-large"

    def _get_backup_model(self) -> str:
        """Get backup model name from config, falling back to default."""
        return getattr(self.config, "backup_model_name_or_path", None) or "text-embedding-3-large"

    @timed_with_status(
        log_prefix="model_timed_embedding",
        log_extra_args=lambda self, texts: {
            "model_name_or_path": self._get_model(),
            "text_len": len(texts),
            "text_content": texts,
        },
    )
    def embed(self, texts: list[str]) -> list[list[float]]:
        if isinstance(texts, str):
            texts = [texts]
        # Sanitize Unicode to prevent encoding errors with emoji/surrogates
        texts = [_sanitize_unicode(t) for t in texts]
        # Truncate texts if max_tokens is configured
        texts = self._truncate_texts(texts)
        logger.info(f"Embeddings request with input: {texts}")
        if self.provider == "openai" or self.provider == "azure":
            try:
                init_time = time.time()
                # Use sync OpenAI client directly — no asyncio.run() needed.
                # asyncio.run() fails when called from within an existing event loop
                # (e.g. uvicorn, the standard FastAPI ASGI server).
                # The openai.OpenAI client is synchronous by design.
                model = self._get_model()
                response = self.client.embeddings.create(
                    model=model,
                    input=texts,
                    timeout=int(os.getenv("MOS_EMBEDDER_TIMEOUT", 30)),
                )
                logger.info(f"Embeddings request succeeded with {time.time() - init_time} seconds")
                return [r.embedding for r in response.data]
            except Exception as e:
                if self.use_backup_client:
                    logger.warning(
                        f"Embeddings request ended with {type(e).__name__} error: {e}, try backup client"
                    )
                    try:
                        init_time = time.time()
                        backup_model = self._get_backup_model()
                        response = self.backup_client.embeddings.create(
                            model=backup_model,
                            input=texts,
                            timeout=int(os.getenv("MOS_EMBEDDER_TIMEOUT", 30)),
                        )
                        logger.info(
                            f"Backup embeddings request succeeded with {time.time() - init_time} seconds"
                        )
                        logger.info(f"Backup embeddings request response: {response}")
                        return [r.embedding for r in response.data]
                    except Exception as e:
                        raise ValueError(f"Backup embeddings request ended with error: {e}") from e
                else:
                    raise ValueError(f"Embeddings request ended with error: {e}") from e
        else:
            raise ValueError(f"Embeddings unsupported provider: {self.provider}")
