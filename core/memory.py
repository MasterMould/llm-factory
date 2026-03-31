# ---------------------------------------------------------------------------
# Memory Management System with ChromaDB REST Client
# ---------------------------------------------------------------------------
import logging
from typing import Dict, List, Optional, Union
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from requests_ratelimiter import LimiterSession

from core.config import (
    MEMORY_URL,
    RETRY_ATTEMPTS,
    RETRY_BACKOFF,
    RETRY_STATUS_CODES,
    MAX_REQUESTS_PER_MINUTE,
    API_KEY
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DEFAULT_TIMEOUT = (5, 15)  # (connect, read)
DEBUG_MEMORY = False


# ---------------------------------------------------------------------------
# HTTP session with retries + rate limiting
# ---------------------------------------------------------------------------
def _build_session(api_key: str = API_KEY) -> requests.Session:
    session = LimiterSession(per_minute=MAX_REQUESTS_PER_MINUTE)

    retries = Retry(
        total=RETRY_ATTEMPTS,
        backoff_factor=RETRY_BACKOFF,
        status_forcelist=RETRY_STATUS_CODES,
        allowed_methods=["GET", "POST", "DELETE"]
    )

    adapter = HTTPAdapter(max_retries=retries)
    session.mount("http://", adapter)
    session.mount("https://", adapter)

    session.headers.update({
        "X-API-Key": api_key,
        "User-Agent": "MemoryManager/2.0"
    })

    return session


_session = _build_session()


# ---------------------------------------------------------------------------
# Custom Exceptions
# ---------------------------------------------------------------------------
class MemoryManagerError(Exception):
    pass


class AuthenticationError(MemoryManagerError):
    pass


class RateLimitError(MemoryManagerError):
    pass


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------
def _handle_response(r: requests.Response, context: str):
    if DEBUG_MEMORY:
        logger.info("📡 %s → %s", context, r.status_code)

    if r.status_code == 401:
        raise AuthenticationError(f"Unauthorized access: {context}")

    if r.status_code == 429:
        raise RateLimitError(f"Rate limit exceeded: {context}")

    try:
        r.raise_for_status()
    except Exception as e:
        logger.exception("❌ HTTP error during %s", context)
        raise MemoryManagerError(str(e)) from e

    try:
        return r.json()
    except Exception as e:
        logger.exception("❌ JSON decode failed during %s", context)
        raise MemoryManagerError("Invalid JSON response") from e


def _normalize_memories(data) -> List[Dict]:
    memories = data.get("memories", data) if isinstance(data, dict) else data

    if not isinstance(memories, list):
        logger.warning("⚠️ Unexpected memory format: %s", type(memories))
        return []

    seen = set()
    filtered = []

    for m in memories:
        key = (
            m.get("id")
            or m.get("memory")
            or m.get("text")
            or str(m)
        )

        if key not in seen:
            seen.add(key)
            filtered.append(m)

    return filtered


# ---------------------------------------------------------------------------
# MemoryManager
# ---------------------------------------------------------------------------
class MemoryManager:

    # -----------------------------------------------------------------------
    # STORE MEMORY
    # -----------------------------------------------------------------------
    @staticmethod
    def memorize(
        messages: List[Dict],
        user_id: str,
        session_id: Optional[str] = None
    ) -> Dict:
        url = f"{MEMORY_URL}/memorize"
        payload = {
            "messages": messages,
            "user_id": user_id,
            "session_id": session_id
        }

        try:
            if DEBUG_MEMORY:
                logger.info("📡 POST %s | payload=%s", url, payload)

            r = _session.post(url, json=payload, timeout=DEFAULT_TIMEOUT)
            return _handle_response(r, "memorize")

        except Exception as e:
            logger.exception("❌ Memory memorize failed")
            raise MemoryManagerError(str(e)) from e

    # -----------------------------------------------------------------------
    # RETRIEVE MEMORY
    # -----------------------------------------------------------------------
    @staticmethod
    def retrieve(
        query: str,
        user_id: str,
        top_k: int = 5,
        session_id: Optional[str] = None
    ) -> List[Dict]:
        url = f"{MEMORY_URL}/retrieve"
        payload = {
            "query": query,
            "user_id": user_id,
            "top_k": top_k,
            "session_id": session_id
        }

        try:
            if DEBUG_MEMORY:
                logger.info("📡 POST %s | payload=%s", url, payload)

            r = _session.post(url, json=payload, timeout=DEFAULT_TIMEOUT)
            data = _handle_response(r, "retrieve")

            return _normalize_memories(data)

        except Exception as e:
            logger.exception("❌ Memory retrieve failed")
            raise MemoryManagerError(str(e)) from e

    # -----------------------------------------------------------------------
    # LIST ALL
    # -----------------------------------------------------------------------
    @staticmethod
    def list_all(
        user_id: str,
        session_id: Optional[str] = None
    ) -> List[Dict]:
        url = f"{MEMORY_URL}/memories"
        params = {
            "user_id": user_id,
            "session_id": session_id
        }

        try:
            if DEBUG_MEMORY:
                logger.info("📡 GET %s | params=%s", url, params)

            r = _session.get(url, params=params, timeout=DEFAULT_TIMEOUT)
            data = _handle_response(r, "list_all")

            return _normalize_memories(data)

        except Exception as e:
            logger.exception("❌ Memory list_all failed")
            raise MemoryManagerError(str(e)) from e

    # -----------------------------------------------------------------------
    # DELETE
    # -----------------------------------------------------------------------
    @staticmethod
    def delete(
        memory_id: str,
        user_id: str,
        session_id: Optional[str] = None
    ) -> bool:
        url = f"{MEMORY_URL}/memories/{memory_id}"
        params = {
            "user_id": user_id,
            "session_id": session_id
        }

        try:
            if DEBUG_MEMORY:
                logger.info("📡 DELETE %s | params=%s", url, params)

            r = _session.delete(url, params=params, timeout=DEFAULT_TIMEOUT)

            if r.status_code == 401:
                raise AuthenticationError("Unauthorized delete")

            if r.status_code == 429:
                raise RateLimitError("Rate limit exceeded on delete")

            if r.status_code in (200, 204):
                return True

            logger.warning("⚠️ Delete failed: %s", r.status_code)
            return False

        except Exception as e:
            logger.exception("❌ Memory delete failed")
            raise MemoryManagerError(str(e)) from e

    # -----------------------------------------------------------------------
    # BUILD CONTEXT
    # -----------------------------------------------------------------------
    @staticmethod
    def build_context(
        query: str,
        user_id: str,
        top_k: int = 5,
        max_chars: int = 1000,
        session_id: Optional[str] = None
    ) -> str:
        try:
            memories = MemoryManager.retrieve(
                query=query,
                user_id=user_id,
                top_k=top_k,
                session_id=session_id
            )

            if not memories:
                return ""

            seen = set()
            lines = ["## Relevant memories about this user:"]
            total_chars = 0

            for m in memories:
                text = (
                    m.get("memory")
                    or m.get("text")
                    or str(m)
                ).strip()

                if text in seen:
                    continue
                seen.add(text)

                score = m.get("score")
                prefix = f"[{round(score, 3)}] " if score else ""

                line = f"- {prefix}{text}"
                line_len = len(line) + 1

                if total_chars + line_len > max_chars:
                    break

                lines.append(line)
                total_chars += line_len

            return "\n".join(lines)

        except Exception as e:
            logger.exception("❌ Context build failed")
            raise MemoryManagerError(str(e)) from e


# ---------------------------------------------------------------------------
# Example Usage / Smoke Test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    try:
        user_id = "test_user"

        print("=== Memorize ===")
        MemoryManager.memorize(
            [{"role": "user", "content": "I like fast Rust systems"}],
            user_id=user_id
        )

        print("\n=== Retrieve ===")
        mems = MemoryManager.retrieve("What do I like?", user_id)
        print(mems)

        print("\n=== Context ===")
        ctx = MemoryManager.build_context("What do I like?", user_id)
        print(ctx)

        print("\n=== List ===")
        all_mems = MemoryManager.list_all(user_id)
        print(all_mems)

        if all_mems:
            mid = all_mems[0].get("id")
            if mid:
                print("\n=== Delete ===")
                print(MemoryManager.delete(mid, user_id))

    except Exception as e:
        print("🔥 Error:", e)
