import hashlib
import time
from dataclasses import dataclass
from pathlib import Path

import httpx
from selectolax.parser import HTMLParser
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

_FTS_URL = "https://efts.sec.gov/LATEST/search-index"


@dataclass
class SECFiling:
    accession_no: str
    title: str
    form_type: str
    filed_date: str
    doc_url: str | None = None


class SECClient:
    """Live SEC EDGAR access with on-disk cache and rate limiting (~10 req/s)."""

    def __init__(self, user_agent: str, cache_dir: Path | None = None, *, min_interval: float = 0.12) -> None:
        self._headers = {"User-Agent": user_agent}
        self._cache_dir = Path(cache_dir) if cache_dir else None
        if self._cache_dir:
            self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._min_interval = min_interval
        self._last_call = 0.0

    @staticmethod
    def html_to_text(html: str) -> str:
        tree = HTMLParser(html)
        for tag in tree.css("script, style"):
            tag.decompose()
        body = tree.body or tree
        text = body.text(separator="\n")
        lines = [ln.strip() for ln in text.splitlines()]
        return "\n".join(ln for ln in lines if ln)

    def _throttle(self) -> None:
        elapsed = time.monotonic() - self._last_call
        if elapsed < self._min_interval:
            time.sleep(self._min_interval - elapsed)
        self._last_call = time.monotonic()

    def _cache_path(self, url: str) -> Path | None:
        if not self._cache_dir:
            return None
        return self._cache_dir / (hashlib.sha256(url.encode()).hexdigest() + ".txt")

    @retry(
        retry=retry_if_exception_type((httpx.HTTPError,)),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=0.5, max=8),
        reraise=True,
    )
    def _get(self, url: str, **kwargs):
        self._throttle()
        with httpx.Client(timeout=30.0, headers=self._headers, follow_redirects=True) as client:
            resp = client.get(url, **kwargs)
            resp.raise_for_status()
            return resp

    def fetch_document(self, url: str) -> str:
        cache = self._cache_path(url)
        if cache and cache.exists():
            return cache.read_text()
        text = self.html_to_text(self._get(url).text)
        if cache:
            cache.write_text(text)
        return text

    def full_text_search(self, query: str, *, forms: list[str] | None = None, limit: int = 10) -> list[SECFiling]:
        params = {"q": query, "from": 0}
        if forms:
            params["forms"] = ",".join(forms)
        data = self._get(_FTS_URL, params=params).json()
        hits = data.get("hits", {}).get("hits", [])[:limit]
        out: list[SECFiling] = []
        for h in hits:
            src = h.get("_source", {})
            names = src.get("display_names") or ["Unknown"]
            out.append(
                SECFiling(
                    accession_no=h.get("_id", "").split(":")[0],
                    title=names[0],
                    form_type=src.get("form", ""),
                    filed_date=src.get("file_date", ""),
                )
            )
        return out
