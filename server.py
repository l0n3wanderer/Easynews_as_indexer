"""
Production-ready server module for an Easynews-style indexer with Anime category support.

Features:
- FastAPI-based HTTP API (Newznab-like endpoints)
- Pydantic settings for configuration via environment variables
- Robust categorization heuristics for Movies, TV, and Anime (including HD/UHD)
- Structured logging, input validation, API key protection
- Minimal in-memory search backend so endpoints are runnable as a standalone demo

Notes for production deployment:
- Replace the in-memory `sample_items` and `search_backend` with your real Easynews client.
- Consider adding authentication, rate-limiting, monitoring, metrics, and request tracing.
- Run with an ASGI server (uvicorn/gunicorn+uvicorn workers) behind a reverse proxy.
"""

import os
import re
import logging
import uuid
from typing import List, Optional, Dict, Any
from enum import Enum

from fastapi import FastAPI, Query, HTTPException, Request, status, Depends
from fastapi.responses import JSONResponse
from pydantic import BaseModel, BaseSettings, Field
from starlette.middleware.cors import CORSMiddleware

# ----------------------------
# Configuration
# ----------------------------

class Settings(BaseSettings):
    NEWZNAB_APIKEY: str = Field(..., env="NEWZNAB_APIKEY")
    PORT: int = Field(8081, env="PORT")
    STRICT_MATCHING: bool = Field(True, env="STRICT_MATCHING")
    DEFAULT_LIMIT: int = Field(100, env="DEFAULT_LIMIT")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()

# ----------------------------
# Logging
# ----------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
logger = logging.getLogger("easynews_indexer")

# ----------------------------
# Constants: Newznab categories
# ----------------------------

CATEGORY_MOVIES = 2000
CATEGORY_MOVIES_HD = 2030
CATEGORY_MOVIES_UHD = 2040
CATEGORY_TV = 5000
CATEGORY_TV_HD = 5030
CATEGORY_TV_UHD = 5040

# Anime categories (added)
CATEGORY_ANIME = 5070
CATEGORY_ANIME_HD = 5075

CATEGORY_OTHER = 7000

# ----------------------------
# Regexes / heuristics
# ----------------------------

_QUALITY_RE = re.compile(r"\b(2160|4k|uhd|1440|1080|720|480|360)\b", re.IGNORECASE)
_SEASON_EP_RE = re.compile(
    r"(?:s(?P<s>\d{1,2})e(?P<e>\d{1,2})|(?P<s2>\d{1,2})x(?P<e2>\d{1,2})|ep[.\s-]?(?P<ep>\d{1,3}))",
    re.IGNORECASE,
)
_ANIME_KEYWORDS = re.compile(
    r"\b(anime|ova|ova|subbed|dubbed|raw|hardsub|vostfr|engsub|eng-sub|bd|bdrip|bdr|bluray)\b",
    re.IGNORECASE,
)
_JAPANESE_CHAR_RE = re.compile(
    r"[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff]"
)  # Hiragana, Katakana, Kanji
_TOKEN_SPLIT_RE = re.compile(r"[^\w]+", re.UNICODE)

# ----------------------------
# Models
# ----------------------------

class ContentType(str, Enum):
    search = "search"
    movie = "movie"
    tvsearch = "tvsearch"
    anime = "anime"


class Item(BaseModel):
    id: str
    title: str
    size: int  # in MB for demo
    category: int
    quality: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


# ----------------------------
# Categorization logic
# ----------------------------

def detect_quality(title: str) -> Optional[str]:
    """
    Return a normalized quality string: 'uhd', 'hd', 'sd', or None.
    Uses common numeric tags and 4K/2160/1080/720/etc.
    """
    m = _QUALITY_RE.search(title)
    if not m:
        return None
    q = m.group(1).lower()
    if q in {"2160", "4k", "uhd"}:
        return "uhd"
    if q in {"1080", "720", "1440"}:
        return "hd"
    # fallback
    return "sd"


def looks_like_anime(title: str) -> bool:
    """
    Heuristic checks for anime:
    - explicit 'anime' token or known anime-related keywords (BD, OVA, subbed, raw, etc.)
    - presence of Japanese characters (hiragana/katakana/kanji)
    - release patterns common to anime (often 'BD', 'BDrip', or 'Raw' markers)
    """
    if _ANIME_KEYWORDS.search(title):
        return True
    if _JAPANESE_CHAR_RE.search(title):
        return True
    # some releases include 'TV' but are anime; presence of 'BD' or 'BDrip' is a useful hint
    if re.search(r"\b(bd|bdrip|bluray|h264|x264|x265)\b", title, re.IGNORECASE) and "season" not in title.lower():
        # treat as possible anime/movie BluRay; additional heuristics needed for real backend
        return True
    return False


def looks_like_tv(title: str) -> bool:
    """
    Detect TV by presence of SxxExx, 'season', 'episode', or common TV tokens.
    """
    if _SEASON_EP_RE.search(title):
        return True
    if re.search(r"\b(season|episode|ep)\b", title, re.IGNORECASE):
        return True
    return False


def categorize_item(title: str) -> int:
    """
    Assign a Newznab category ID for the given title.
    Rules implemented:
      - Anime detection (anime keywords or Japanese characters)
      - TV detection (SxxExx or season/episode tokens)
      - Movies otherwise (fallback)
      - Quality modifiers map to HD / UHD categories where appropriate
    """
    if not title:
        return CATEGORY_OTHER
    t = title.lower()

    quality = detect_quality(t)

    # Anime detection has priority over generic TV detection because anime releases
    # often include "S01E.." but are better grouped under anime for clients that
    # treat anime separately.
    if looks_like_anime(t):
        if quality == "uhd":
            return CATEGORY_ANIME_HD  # treat UHD anime as HD-class; adjust if needed
        if quality == "hd":
            return CATEGORY_ANIME_HD
        return CATEGORY_ANIME

    # TV detection (non-anime)
    if looks_like_tv(t):
        if quality == "uhd":
            return CATEGORY_TV_UHD
        if quality == "hd":
            return CATEGORY_TV_HD
        return CATEGORY_TV

    # Movies
    # If title contains 'movie' or typical movie markers, classify as movie
    if re.search(r"\b(movie|film|feature)\b", t) or re.search(r"\b(1080p|720p|uhd|bluray|bdrip)\b", t):
        if quality == "uhd":
            return CATEGORY_MOVIES_UHD
        if quality == "hd":
            return CATEGORY_MOVIES_HD
        return CATEGORY_MOVIES

    # Default
    return CATEGORY_OTHER


# ----------------------------
# Simple in-memory "search backend" for demo / tests
# ----------------------------

sample_items = [
    {"title": "Naruto Shippuden S01E01 1080p BD (JP) [Anime]", "size": 1200},
    {"title": "Dragon Ball Z 720p Bluray", "size": 900},
    {"title": "Friends S01E01 720p", "size": 800},
    {"title": "Inception Movie UHD 2160p", "size": 2500},
    {"title": "My Documentary 480p", "size": 600},
    {"title": "進撃の巨人 S01E01 1080p BD", "size": 1100},  # Contains Japanese chars
]


def search_backend(query: str, limit: int) -> List[Item]:
    """
    Simple token-based filter. In production, this should query the real Easynews API
    and apply stricter matching, paging, and sorting.
    """
    if not query:
        return []

    qtokens = {tok for tok in _TOKEN_SPLIT_RE.sub(" ", query.lower()).split() if tok}

    results: List[Item] = []
    for raw in sample_items:
        title = raw["title"]
        t_tokens = set(_TOKEN_SPLIT_RE.sub(" ", title.lower()).split())
        # strict match: all query tokens present
        if settings.STRICT_MATCHING:
            if not qtokens.issubset(t_tokens):
                continue
        else:
            if not (qtokens & t_tokens):
                continue

        category = categorize_item(title)
        quality = detect_quality(title)
        item = Item(
            id=str(uuid.uuid5(uuid.NAMESPACE_URL, title)),
            title=title,
            size=raw["size"],
            category=category,
            quality=quality,
            metadata={},
        )
        results.append(item)
        if len(results) >= limit:
            break

    return results


# ----------------------------
# FastAPI app and endpoints
# ----------------------------

app = FastAPI(title="Easynews AS Indexer (demo)", version="1.0.0")

# Simple CORS -- tune origins in production
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


def validate_apikey(apikey: str = Query(..., alias="apikey")):
    if apikey != settings.NEWZNAB_APIKEY:
        logger.warning("Unauthorized access attempt with apikey=%s", apikey)
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid API key")
    return True


@app.get("/api", response_model=Dict[str, Any])
def api_endpoint(
    t: ContentType = Query(ContentType.search, description="API action/type"),
    q: Optional[str] = Query(None, description="Search query"),
    limit: Optional[int] = Query(None, ge=1, le=1000),
    apikey_ok: bool = Depends(validate_apikey),
    request: Request = None,
):
    """
    Minimal Newznab-like API:
      - t=search / t=movie / t=tvsearch / t=anime
    Returns JSON with a simple 'items' list; replace with XML/Newznab response
    if needed for compatibility.
    """
    limit = limit or settings.DEFAULT_LIMIT

    if t in {ContentType.search, ContentType.movie, ContentType.tvsearch, ContentType.anime}:
        if not q:
            raise HTTPException(status_code=400, detail="q (query) parameter is required for search endpoints")
        # For demo: map t to internal behavior; production will call Easynews
        results = search_backend(q, limit)
        # Optionally filter by requested type
        if t == ContentType.movie:
            results = [r for r in results if r.category in {CATEGORY_MOVIES, CATEGORY_MOVIES_HD, CATEGORY_MOVIES_UHD}]
        elif t == ContentType.tvsearch:
            results = [r for r in results if r.category in {CATEGORY_TV, CATEGORY_TV_HD, CATEGORY_TV_UHD}]
        elif t == ContentType.anime:
            results = [r for r in results if r.category in {CATEGORY_ANIME, CATEGORY_ANIME_HD}]
        # Serialize
        return {
            "query": q,
            "t": t.value,
            "count": len(results),
            "items": [r.dict() for r in results],
        }

    # fallback / not implemented
    raise HTTPException(status_code=400, detail=f"Unsupported type '{t}'")


@app.get("/api/get", response_model=Dict[str, Any])
def api_get(id: str = Query(..., description="Item id"), apikey_ok: bool = Depends(validate_apikey)):
    """
    Return details for a given item id. Production implementation should stream or proxy
    an NZB or redirect to download location.
    """
    for raw in sample_items:
        candidate_id = str(uuid.uuid5(uuid.NAMESPACE_URL, raw["title"]))
        if candidate_id == id:
            category = categorize_item(raw["title"])
            return {"id": id, "title": raw["title"], "size": raw["size"], "category": category}
    raise HTTPException(status_code=404, detail="Item not found")


@app.get("/api/tags", response_model=Dict[str, Any])
def api_tags(apikey_ok: bool = Depends(validate_apikey)):
    """
    Return category map and other discovery info for clients.
    """
    return {
        "categories": {
            "movies": [CATEGORY_MOVIES, CATEGORY_MOVIES_HD, CATEGORY_MOVIES_UHD],
            "tv": [CATEGORY_TV, CATEGORY_TV_HD, CATEGORY_TV_UHD],
            "anime": [CATEGORY_ANIME, CATEGORY_ANIME_HD],
            "other": [CATEGORY_OTHER],
        }
    }


# ----------------------------
# Error handlers
# ----------------------------

@app.exception_handler(HTTPException)
def http_exception_handler(request: Request, exc: HTTPException):
    logger.info("HTTPException: %s %s", exc.status_code, exc.detail)
    return JSONResponse({"error": exc.detail}, status_code=exc.status_code)


@app.exception_handler(Exception)
def general_exception_handler(request: Request, exc: Exception):
    logger.exception("Unhandled exception: %s", exc)
    return JSONResponse({"error": "internal server error"}, status_code=500)


# ----------------------------
# Entrypoint
# ----------------------------

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("server:app", host="0.0.0.0", port=settings.PORT, log_level="info")