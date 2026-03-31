# core/memory.py — MemoryManager: mem0 + ChromaDB REST client (:8000)

import logging
from typing import Dict, List, Any, Optional
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from core.config import MEMORY_URL

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# HTTP session with retries (resilient networking)
# ---------------------------------------------------------------------------
def _build_session() -> requests.Session:
    session = requests.Session()
    retries = Retry(
        total=3,
        backoff_factor=0.5,
        status_forcelist=[500, 502, 503, 504],
        allowed_methods=["GET", "POST", "DELETE"]
    )
    adapter = HTTPAdapter(max_retries=retries)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session


_session = _build_session()


class MemoryManager:

    # -----------------------------------------------------------------------
    # STORE MEMORY
    # -----------------------------------------------------------------------
    @staticmethod
    def memorize(messages: List[Dict], user_id: str) -> Dict:
        try:
            r = _session.post(
                f"{MEMORY_URL}/memorize",
                json={"messages": messages, "user_id": user_id},
                timeout=30
            )
            r.raise_for_status()
            return r.json()
        except Exception as e:
            logger.exception("❌ Memory memorize failed")
            return {"error": str(e)}

    # -----------------------------------------------------------------------
    # RETRIEVE MEMORY (semantic search)
    # -----------------------------------------------------------------------
    @staticmethod
    def retrieve(query: str, user_id: str, top_k: int = 5) -> List[Dict]:
        try:
            r = _session.post(
                f"{MEMORY_URL}/retrieve",
                json={"query": query, "user_id": user_id, "top_k": top_k},
                timeout=15
            )
            r.raise_for_status()
            data = r.json()

            memories = data.get("memories", data) if isinstance(data, dict) else data

            # Defensive normalization
            if not isinstance(memories, list):
                logger.warning("⚠️ Unexpected memory format: %s", type(memories))
                return []

            return memories

        except Exception:
            logger.exception("❌ Memory retrieve failed")
            return []

    # -----------------------------------------------------------------------
    # LIST ALL MEMORIES
    # -----------------------------------------------------------------------
    @staticmethod
    def list_all(user_id: str) -> List[Dict]:
        try:
            r = _session.get(
                f"{MEMORY_URL}/memories",
                params={"user_id": user_id},
                timeout=10
            )
            r.raise_for_status()
            data = r.json()
            return data.get("memories", data) if isinstance(data, dict) else data
        except Exception:
            logger.exception("❌ Memory list_all failed")
            return []

    # -----------------------------------------------------------------------
    # DELETE MEMORY
    # -----------------------------------------------------------------------
    @staticmethod
    def delete(memory_id: str, user_id: str) -> bool:
        try:
            r = _session.delete(
                f"{MEMORY_URL}/memories/{memory_id}",
                params={"user_id": user_id},
                timeout=10
            )
            if r.status_code not in (200, 204):
                logger.warning("⚠️ Delete failed: %s", r.status_code)
            return r.status_code in (200, 204)
        except Exception:
            logger.exception("❌ Memory delete failed")
            return False

    # -----------------------------------------------------------------------
    # BUILD CONTEXT (LLM-ready string)
    # -----------------------------------------------------------------------
    @staticmethod
    def build_context(
        query: str,
        user_id: str,
        max_items: int = 5,
        max_chars: int = 1000
    ) -> str:

        mems = MemoryManager.retrieve(query, user_id, top_k=max_items)

        if not mems:
            return ""

        seen = set()
        lines = ["## Relevant memories about this user:"]

        total_chars = 0

        for m in mems:
            text = (
                m.get("memory")
                or m.get("text")
                or str(m)
            ).strip()

            # Deduplicate
            if text in seen:
                continue
            seen.add(text)

            # Optional score display
            score = m.get("score")
            prefix = f"[{round(score, 3)}] " if score else ""

            line = f"- {prefix}{text}"

            # Enforce context size limit
            if total_chars + len(line) > max_chars:
                break

            lines.append(line)
            total_chars += len(line)

        return "\n".join(lines)
