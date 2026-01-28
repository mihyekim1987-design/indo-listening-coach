# api/youtube_transcripts.py
from __future__ import annotations

import re
from urllib.parse import urlparse, parse_qs
from typing import Iterable

def extract_video_id(url_or_id: str) -> str:
    """
    지원:
    - https://www.youtube.com/watch?v=VIDEOID
    - https://youtu.be/VIDEOID
    - https://www.youtube.com/shorts/VIDEOID
    - https://www.youtube.com/embed/VIDEOID
    - VIDEOID (직접 입력)
    """
    s = (url_or_id or "").strip()
    if not s:
        raise ValueError("Empty YouTube url/id")

    # video id 직접 입력 (보통 11자)
    if re.fullmatch(r"[A-Za-z0-9_-]{11}", s):
        return s

    u = urlparse(s)
    host = (u.netloc or "").lower()
    path = (u.path or "").strip("/")

    # watch?v=
    if "youtube.com" in host:
        qs = parse_qs(u.query or "")
        v = (qs.get("v") or [None])[0]
        if v and re.fullmatch(r"[A-Za-z0-9_-]{11}", v):
            return v

        # /shorts/<id>, /embed/<id>
        parts = path.split("/")
        if len(parts) >= 2 and parts[0] in {"shorts", "embed"}:
            cand = parts[1]
            if re.fullmatch(r"[A-Za-z0-9_-]{11}", cand):
                return cand

    # youtu.be/<id>
    if "youtu.be" in host:
        parts = path.split("/")
        if parts and re.fullmatch(r"[A-Za-z0-9_-]{11}", parts[0]):
            return parts[0]

    raise ValueError("Could not extract video_id from URL")

def _join_snippets(fetched) -> str:
    texts = []
    for snip in fetched:
        t = getattr(snip, "text", None) or ""
        if t:
            texts.append(t)
    out = " ".join(texts)
    out = re.sub(r"\s+", " ", out).strip()
    return out

def fetch_youtube_transcript(
    url_or_id: str,
    prefer_langs: Iterable[str] = ("id", "en"),
    preserve_formatting: bool = False,
) -> str:
    """
    youtube-transcript-api v1.2.3 기준:
    - YouTubeTranscriptApi().fetch(video_id, languages=[...]) 만 사용
    """
    from youtube_transcript_api import YouTubeTranscriptApi

    video_id = extract_video_id(url_or_id)
    langs = [x for x in prefer_langs if x]

    ytt = YouTubeTranscriptApi()
    fetched = ytt.fetch(video_id, languages=langs, preserve_formatting=preserve_formatting)
    text = _join_snippets(fetched)

    if not text:
        raise RuntimeError("Empty transcript fetched (no captions or blocked).")

    return text
