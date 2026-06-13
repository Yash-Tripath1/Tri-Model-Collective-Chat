#!/usr/bin/env python3
import hashlib
import html
import json
import os
import re
import time
import uuid
from html.parser import HTMLParser
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib import error, request
from urllib.parse import parse_qs, quote, unquote, urlparse

HOST = "0.0.0.0"
PORT = int(os.environ.get("PORT", "8000"))

SESSIONS = {}
MAX_RECENT_TURNS = 6
MAX_CONTEXT_CHARS = 5000
USER_AGENT = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125 Safari/537.36"

DEFAULT_SETTINGS = {
    "mode": "auto",
    "models": {
        "claude": "claude-3-5-haiku-latest",
        "groq": "llama-3.1-8b-instant",
        "gemini": "gemini-2.0-flash",
    },
    "temperature": 0.2,
    "budgets": {
        "planner": 180,
        "research": 260,
        "builder": 620,
        "verifier": 320,
        "summary": 160,
    },
    "web_search": {
        "mode": "off",
        "engine": "duckduckgo_lite",
        "max_results": 4,
        "fetch_pages": 2,
        "page_chars": 700,
    },
    "documents": {
        "max_files_per_request": 4,
        "max_chars_per_file": 20000,
        "chunk_chars": 1200,
        "top_chunks": 4,
        "chunk_excerpt_chars": 700,
    },
    "pricing": {
        "claude": {"input_per_million": 0.0, "output_per_million": 0.0},
        "groq": {"input_per_million": 0.0, "output_per_million": 0.0},
        "gemini": {"input_per_million": 0.0, "output_per_million": 0.0},
    },
}

INDEX_HTML_PATH = "index.html"

PLANNER_SYSTEM = """You are the planner in a multi-model pipeline.
Return STRICT JSON only.
Schema:
{
  \"intent\": \"\",
  \"approach\": [\"\", \"\"],
  \"needs_research\": [\"\", \"\"],
  \"risks\": [\"\", \"\"],
  \"answer_style\": \"\"
}
Rules: concise, no markdown, max 2 items per array, each item under 12 words."""

RESEARCH_SYSTEM = """You are the fast critic/researcher in a multi-model pipeline.
Use provided grounded context if available.
Do NOT claim live web access unless sources are explicitly provided in the prompt.
Return STRICT JSON only.
Schema:
{
  \"facts\": [\"\", \"\"],
  \"assumptions\": [\"\", \"\"],
  \"checks\": [\"\", \"\"],
  \"build_notes\": [\"\", \"\"]
}
Rules: concise, no markdown, max 2 items per array, each item under 16 words."""

BUILDER_SYSTEM = """You are the synthesizer. Build the final response using the user request and peer notes.
Be accurate, direct, and token-efficient.
If grounded sources are provided, use only supported claims and cite them exactly as instructed.
If something is uncertain, say so briefly instead of guessing.
Return the final answer only, with clean formatting."""

VERIFIER_SYSTEM = """You are the final verifier.
Tighten the draft, remove overclaims, and preserve only supported content.
If grounded sources are provided, keep only valid citations and do not invent any.
Keep it concise and useful.
Return the final answer only."""

SUMMARY_SYSTEM = """Compress the conversation into a short memory note.
Return 4-6 bullets, each under 14 words.
Keep only durable goals, decisions, constraints, and unresolved items."""


class VisibleTextExtractor(HTMLParser):
    def __init__(self):
        super().__init__()
        self.parts = []
        self.skip_depth = 0
        self.skip_tags = {"script", "style", "noscript", "svg"}
        self.block_tags = {
            "p", "div", "section", "article", "main", "header", "footer", "aside",
            "h1", "h2", "h3", "h4", "h5", "h6", "li", "ul", "ol", "table", "tr", "td", "br"
        }

    def handle_starttag(self, tag, attrs):
        if tag in self.skip_tags:
            self.skip_depth += 1
            return
        if self.skip_depth == 0 and tag in self.block_tags:
            self.parts.append("\n")

    def handle_endtag(self, tag):
        if tag in self.skip_tags and self.skip_depth > 0:
            self.skip_depth -= 1
            return
        if self.skip_depth == 0 and tag in self.block_tags:
            self.parts.append("\n")

    def handle_data(self, data):
        if self.skip_depth == 0 and data:
            self.parts.append(data)


def now_ts():
    return int(time.time())


def json_clone(value):
    return json.loads(json.dumps(value))


def deep_merge(base, override):
    result = json_clone(base)
    if not isinstance(override, dict):
        return result
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def get_api_keys(payload):
    api_keys = payload.get("apiKeys") or {}
    return {
        "claude": api_keys.get("claude") or os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("CLAUDE_API_KEY") or "",
        "groq": api_keys.get("groq") or os.environ.get("GROQ_API_KEY") or "",
        "gemini": api_keys.get("gemini") or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY") or "",
    }


def provider_available(api_keys, provider):
    return bool((api_keys.get(provider) or "").strip())


def rough_tokens(text):
    return max(1, len((text or "").strip()) // 4)


def safe_float(value, default=0.0):
    try:
        return float(value)
    except Exception:
        return default


def trim_text(text, max_chars):
    text = (text or "").strip()
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3] + "..."


def normalize_space(text):
    return re.sub(r"\s+", " ", (text or "").strip())


def html_to_text(raw_html):
    parser = VisibleTextExtractor()
    parser.feed(raw_html or "")
    parser.close()
    text = html.unescape("".join(parser.parts))
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def clean_fragment(fragment):
    text = re.sub(r"<[^>]+>", " ", fragment or "")
    text = html.unescape(text)
    return normalize_space(text)


def render_turns(turns):
    lines = []
    for turn in turns:
        role = turn.get("role", "user").upper()
        content = trim_text(turn.get("content", ""), 1200)
        lines.append(f"{role}: {content}")
    return "\n\n".join(lines)


def build_context_pack(session):
    summary = (session.get("summary") or "").strip()
    recent_turns = session.get("turns", [])[-MAX_RECENT_TURNS:]
    recent_text = render_turns(recent_turns)
    parts = []
    if summary:
        parts.append("Conversation summary:\n" + trim_text(summary, 1600))
    if recent_text:
        parts.append("Recent turns:\n" + trim_text(recent_text, 3000))
    packed = "\n\n".join(parts).strip()
    return trim_text(packed, MAX_CONTEXT_CHARS)


def choose_route(message, mode):
    if mode == "groq_only":
        return "groq_only"
    if mode == "triad":
        return "triad"
    if mode == "collective":
        return "collective"

    text = (message or "").lower()
    score = 0
    if len(message) > 350:
        score += 1
    if len(message) > 900:
        score += 1
    if message.count("?") >= 2:
        score += 1
    if any(k in text for k in [
        "build", "design", "architecture", "system", "debug", "plan", "compare",
        "tradeoff", "code", "app", "platform", "workflow", "agent"
    ]):
        score += 2
    if any(k in text for k in ["medical", "legal", "security", "finance", "tax", "contract", "production"]):
        return "collective"
    if score <= 1:
        return "groq_only"
    if score <= 3:
        return "triad"
    return "collective"


def extract_terms(text):
    return [t for t in re.findall(r"[a-z0-9]{3,}", (text or "").lower()) if t not in {
        "the", "and", "for", "with", "that", "this", "from", "have", "your", "what",
        "when", "where", "which", "about", "into", "while", "should", "would", "could",
        "please", "make", "build", "using", "there", "their", "them", "they", "want",
        "need", "then", "than", "will", "just", "into", "also", "some", "more"
    }]


def should_use_web_search(message, settings):
    cfg = settings.get("web_search") or {}
    mode = (cfg.get("mode") or "off").lower()
    if mode == "on":
        return True
    if mode == "off":
        return False
    text = (message or "").lower()
    triggers = [
        "latest", "recent", "today", "current", "news", "update", "updated", "new",
        "price", "pricing", "release", "launched", "announcement", "2024", "2025", "2026",
        "this year", "benchmark", "ranking", "who won", "version", "docs", "documentation"
    ]
    return any(token in text for token in triggers)


def decode_duckduckgo_href(href):
    href = html.unescape((href or "").strip())
    if href.startswith("//"):
        href = "https:" + href
    if href.startswith("/"):
        href = "https://duckduckgo.com" + href
    parsed = urlparse(href)
    params = parse_qs(parsed.query)
    if "uddg" in params and params["uddg"]:
        return unquote(params["uddg"][0])
    return href


def http_post_json(url, headers, payload, timeout=90):
    body = json.dumps(payload).encode("utf-8")
    req = request.Request(url, data=body, headers=headers, method="POST")
    try:
        with request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8")
            return json.loads(raw)
    except error.HTTPError as e:
        detail = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {e.code}: {detail[:1200]}")
    except error.URLError as e:
        raise RuntimeError(f"Network error: {e}")


def http_get_text(url, headers=None, timeout=25):
    req = request.Request(url, headers=headers or {}, method="GET")
    try:
        with request.urlopen(req, timeout=timeout) as resp:
            content_type = resp.headers.get("Content-Type", "")
            raw = resp.read()
            charset = "utf-8"
            match = re.search(r"charset=([\w\-]+)", content_type, re.I)
            if match:
                charset = match.group(1)
            return raw.decode(charset, errors="replace"), content_type
    except error.HTTPError as e:
        detail = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {e.code}: {detail[:500]}")
    except error.URLError as e:
        raise RuntimeError(f"Network error: {e}")


def search_duckduckgo_lite(query, max_results=4):
    url = f"https://lite.duckduckgo.com/lite/?q={quote(query)}"
    body, _ = http_get_text(url, headers={"User-Agent": USER_AGENT, "Accept-Language": "en-US,en;q=0.9"}, timeout=25)
    pattern = re.compile(
        r"<a[^>]*href=\"(?P<href>.*?)\"[^>]*class=['\"]result-link['\"][^>]*>(?P<title>.*?)</a>"
        r".*?<td[^>]*class=['\"]result-snippet['\"][^>]*>\s*(?P<snippet>.*?)\s*</td>"
        r".*?<span[^>]*class=['\"]link-text['\"][^>]*>(?P<display>.*?)</span>",
        re.S | re.I,
    )
    results = []
    seen = set()
    for match in pattern.finditer(body):
        real_url = decode_duckduckgo_href(match.group("href"))
        title = clean_fragment(match.group("title"))
        snippet = clean_fragment(match.group("snippet"))
        display = clean_fragment(match.group("display"))
        if not real_url or real_url in seen:
            continue
        seen.add(real_url)
        results.append({
            "title": title or display or real_url,
            "url": real_url,
            "snippet": snippet,
            "display_url": display,
        })
        if len(results) >= max_results:
            break
    return results


def fetch_page_excerpt(url, max_chars=700):
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"}:
        return None
    body, content_type = http_get_text(url, headers={"User-Agent": USER_AGENT, "Accept-Language": "en-US,en;q=0.9"}, timeout=20)
    if "html" not in content_type.lower() and "xml" not in content_type.lower():
        return None
    title_match = re.search(r"<title[^>]*>(.*?)</title>", body, re.S | re.I)
    title = clean_fragment(title_match.group(1)) if title_match else ""
    text = html_to_text(body)
    excerpt = trim_text(text, max_chars)
    if not excerpt:
        return None
    return {
        "title": title,
        "excerpt": excerpt,
    }


def build_web_grounding(message, settings):
    if not should_use_web_search(message, settings):
        return {"enabled": False, "query": "", "sources": [], "context": "", "errors": []}

    cfg = settings.get("web_search") or {}
    max_results = max(1, min(6, int(cfg.get("max_results") or 4)))
    fetch_pages = max(0, min(max_results, int(cfg.get("fetch_pages") or 2)))
    page_chars = max(250, min(1600, int(cfg.get("page_chars") or 700)))
    query = trim_text(normalize_space(message), 240)
    errors = []

    try:
        results = search_duckduckgo_lite(query, max_results=max_results)
    except Exception as e:
        return {"enabled": True, "query": query, "sources": [], "context": "", "errors": [str(e)]}

    sources = []
    for idx, item in enumerate(results, start=1):
        source = {
            "id": idx,
            "title": item.get("title") or item.get("display_url") or item.get("url"),
            "url": item.get("url") or "",
            "search_snippet": item.get("snippet") or "",
            "excerpt": item.get("snippet") or "",
        }
        sources.append(source)

    for source in sources[:fetch_pages]:
        try:
            page = fetch_page_excerpt(source["url"], max_chars=page_chars)
            if page:
                source["title"] = page.get("title") or source["title"]
                source["excerpt"] = page.get("excerpt") or source["excerpt"]
        except Exception as e:
            errors.append(f"{source['url']}: {e}")

    context_lines = []
    for source in sources:
        context_lines.append(
            f"[{source['id']}] {trim_text(source['title'], 140)}\n"
            f"URL: {source['url']}\n"
            f"Evidence: {trim_text(source.get('excerpt') or source.get('search_snippet') or '', page_chars)}"
        )
    context = "\n\n".join(context_lines)
    return {"enabled": True, "query": query, "sources": sources, "context": context, "errors": errors}


def split_text_chunks(text, chunk_chars=1200):
    text = (text or "").replace("\r\n", "\n").replace("\r", "\n").strip()
    if not text:
        return []
    blocks = re.split(r"\n\s*\n", text)
    chunks = []
    current = ""
    for block in blocks:
        block = block.strip()
        if not block:
            continue
        if len(block) > chunk_chars:
            if current:
                chunks.append(current.strip())
                current = ""
            for i in range(0, len(block), chunk_chars):
                piece = block[i:i + chunk_chars].strip()
                if piece:
                    chunks.append(piece)
            continue
        if not current:
            current = block
        elif len(current) + 2 + len(block) <= chunk_chars:
            current += "\n\n" + block
        else:
            chunks.append(current.strip())
            current = block
    if current:
        chunks.append(current.strip())
    return chunks


def ingest_attachments(session, attachments, settings):
    cfg = settings.get("documents") or {}
    max_files = max(1, min(8, int(cfg.get("max_files_per_request") or 4)))
    max_chars = max(500, min(100000, int(cfg.get("max_chars_per_file") or 20000)))
    chunk_chars = max(400, min(4000, int(cfg.get("chunk_chars") or 1200)))
    documents = session.setdefault("documents", {})
    added = []

    for attachment in (attachments or [])[:max_files]:
        name = trim_text((attachment.get("name") or "untitled").replace("\x00", ""), 140)
        content = (attachment.get("content") or "").replace("\x00", "")
        if not content.strip():
            continue
        content = trim_text(content, max_chars)
        doc_id = hashlib.sha1((name + "\n" + content).encode("utf-8", errors="ignore")).hexdigest()[:12]
        if doc_id in documents:
            continue
        chunks = split_text_chunks(content, chunk_chars=chunk_chars)
        if not chunks:
            continue
        documents[doc_id] = {
            "id": doc_id,
            "name": name,
            "size_chars": len(content),
            "type": attachment.get("type") or "text/plain",
            "preview": trim_text(content, 240),
            "chunks": chunks,
            "added_at": now_ts(),
        }
        added.append({
            "id": doc_id,
            "name": name,
            "size_chars": len(content),
            "chunks": len(chunks),
        })
    return added


def score_chunk(message_terms, filename, chunk_text):
    if not chunk_text:
        return 0
    haystack = (filename + "\n" + chunk_text).lower()
    score = 0
    seen = set()
    for term in message_terms:
        if term in seen:
            continue
        seen.add(term)
        if term in haystack:
            score += 3 if term in filename.lower() else 1
    return score


def build_document_grounding(message, session, settings):
    documents = session.get("documents") or {}
    if not documents:
        return {"sources": [], "context": "", "count": 0}

    cfg = settings.get("documents") or {}
    top_chunks = max(1, min(8, int(cfg.get("top_chunks") or 4)))
    excerpt_chars = max(250, min(1800, int(cfg.get("chunk_excerpt_chars") or 700)))
    message_terms = extract_terms(message)
    candidates = []

    for doc in documents.values():
        for idx, chunk in enumerate(doc.get("chunks") or []):
            score = score_chunk(message_terms, doc.get("name", ""), chunk)
            candidates.append({
                "score": score,
                "doc": doc,
                "chunk": chunk,
                "chunk_index": idx,
                "added_at": doc.get("added_at", 0),
            })

    candidates.sort(key=lambda item: (item["score"], item["added_at"]), reverse=True)
    picked = candidates[:top_chunks]
    if not picked or picked[0]["score"] == 0:
        newest_docs = sorted(documents.values(), key=lambda d: d.get("added_at", 0), reverse=True)
        picked = []
        for doc in newest_docs:
            first_chunk = (doc.get("chunks") or [""])[0]
            if first_chunk:
                picked.append({"score": 0, "doc": doc, "chunk": first_chunk, "chunk_index": 0, "added_at": doc.get("added_at", 0)})
            if len(picked) >= top_chunks:
                break

    sources = []
    lines = []
    for idx, item in enumerate(picked, start=1):
        label = f"D{idx}"
        excerpt = trim_text(item["chunk"], excerpt_chars)
        source = {
            "id": label,
            "name": item["doc"].get("name", "document"),
            "chunk_index": item.get("chunk_index", 0),
            "excerpt": excerpt,
        }
        sources.append(source)
        lines.append(f"[{label}] {source['name']}\nExcerpt: {excerpt}")

    return {"sources": sources, "context": "\n\n".join(lines), "count": len(sources)}


def grounding_overview(doc_grounding, web_grounding):
    parts = []
    if doc_grounding.get("count"):
        parts.append(f"{doc_grounding['count']} doc snippets")
    if web_grounding.get("enabled") and web_grounding.get("sources"):
        parts.append(f"{len(web_grounding['sources'])} web sources")
    if not parts:
        return "No external grounding available."
    return "Grounded context available: " + ", ".join(parts) + "."


def build_grounding_text(doc_grounding, web_grounding):
    parts = []
    if doc_grounding.get("context"):
        parts.append("Document context:\n" + doc_grounding["context"])
    if web_grounding.get("context"):
        parts.append("Web sources:\n" + web_grounding["context"])
    return "\n\n".join(parts).strip()


def usage_accumulator():
    return {
        "input_tokens": 0,
        "output_tokens": 0,
        "by_provider": {},
    }


def add_usage(total, provider, usage):
    input_tokens = int((usage or {}).get("input_tokens") or 0)
    output_tokens = int((usage or {}).get("output_tokens") or 0)
    total["input_tokens"] += input_tokens
    total["output_tokens"] += output_tokens
    total["by_provider"].setdefault(provider, {"input_tokens": 0, "output_tokens": 0})
    total["by_provider"][provider]["input_tokens"] += input_tokens
    total["by_provider"][provider]["output_tokens"] += output_tokens


def estimate_cost(usage, settings):
    pricing = (settings or {}).get("pricing") or {}
    total = 0.0
    by_provider = {}
    any_rates = False
    for provider, item in (usage.get("by_provider") or {}).items():
        provider_rates = pricing.get(provider) or {}
        in_rate = max(0.0, safe_float(provider_rates.get("input_per_million"), 0.0))
        out_rate = max(0.0, safe_float(provider_rates.get("output_per_million"), 0.0))
        any_rates = any_rates or bool(in_rate or out_rate)
        input_tokens = int(item.get("input_tokens") or 0)
        output_tokens = int(item.get("output_tokens") or 0)
        input_cost = input_tokens / 1_000_000.0 * in_rate
        output_cost = output_tokens / 1_000_000.0 * out_rate
        subtotal = input_cost + output_cost
        total += subtotal
        by_provider[provider] = {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "input_rate_per_million": in_rate,
            "output_rate_per_million": out_rate,
            "input_cost_usd": round(input_cost, 8),
            "output_cost_usd": round(output_cost, 8),
            "total_cost_usd": round(subtotal, 8),
        }
    return {
        "currency": "USD",
        "available": any_rates,
        "total_cost_usd": round(total, 8),
        "by_provider": by_provider,
        "note": "Set your provider rates in the UI. Pricing changes over time.",
    }


def make_plan_prompt(message, context_pack, overview):
    return (
        f"{context_pack}\n\n{overview}\n\nLatest user request:\n{message}\n\n"
        "Create a compact plan and handoff for other models."
    ).strip()


def make_research_prompt(message, context_pack, plan_text, grounded_text):
    return (
        f"{context_pack}\n\nLatest user request:\n{message}\n\n"
        f"Planner handoff:\n{plan_text}\n\n"
        f"Grounded context:\n{grounded_text or 'None'}\n\n"
        "Critique the plan, note likely facts, assumptions, checks, and useful build notes."
    ).strip()


def make_builder_prompt(message, context_pack, plan_text, research_text, grounded_text):
    return (
        f"{context_pack}\n\nLatest user request:\n{message}\n\n"
        f"Planner handoff:\n{plan_text}\n\n"
        f"Researcher handoff:\n{research_text}\n\n"
        f"Grounded context:\n{grounded_text or 'None'}\n\n"
        "Instructions:\n"
        "- If web sources are provided, cite them inline like [1], [2].\n"
        "- If document snippets are provided, cite them like [D1], [D2].\n"
        "- Do not invent citations.\n"
        "- If evidence is missing, say uncertain briefly.\n\n"
        "Write the best final answer for the user."
    ).strip()


def make_verifier_prompt(message, context_pack, plan_text, research_text, draft_text, grounded_text):
    return (
        f"{context_pack}\n\nLatest user request:\n{message}\n\n"
        f"Planner handoff:\n{plan_text}\n\n"
        f"Researcher handoff:\n{research_text}\n\n"
        f"Grounded context:\n{grounded_text or 'None'}\n\n"
        f"Draft answer:\n{draft_text}\n\n"
        "Instructions:\n"
        "- Keep only supported claims.\n"
        "- Preserve valid citations only.\n"
        "- If citations are unsupported, remove them.\n\n"
        "Verify accuracy, remove overclaims, and return the improved final answer."
    ).strip()


def summarize_session_if_needed(session, api_keys, settings):
    turns = session.get("turns", [])
    if len(turns) <= 10:
        return
    older = turns[:-MAX_RECENT_TURNS]
    if not older:
        return
    transcript = render_turns(older)
    existing_summary = (session.get("summary") or "").strip()
    prompt = (
        f"Existing memory:\n{existing_summary or 'None'}\n\n"
        f"Turns to compress:\n{transcript}\n\n"
        "Create a new compact memory note."
    )
    try:
        result = call_with_fallback(
            ["groq", "gemini", "claude"],
            api_keys,
            settings,
            SUMMARY_SYSTEM,
            prompt,
            settings["budgets"]["summary"],
        )
        session["summary"] = trim_text(result["text"], 1200)
    except Exception:
        session["summary"] = trim_text(existing_summary + "\n" + render_turns(older), 1200)
    session["turns"] = turns[-MAX_RECENT_TURNS:]


def call_claude(api_key, model, system, prompt, max_tokens, temperature):
    payload = {
        "model": model,
        "system": system,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    data = http_post_json(
        "https://api.anthropic.com/v1/messages",
        {
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        },
        payload,
    )
    text = "".join(block.get("text", "") for block in data.get("content", []) if block.get("type") == "text")
    usage = data.get("usage") or {}
    return {
        "text": text.strip(),
        "usage": {
            "input_tokens": usage.get("input_tokens", rough_tokens(system + prompt)),
            "output_tokens": usage.get("output_tokens", rough_tokens(text)),
        },
    }


def call_groq(api_key, model, system, prompt, max_tokens, temperature):
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    data = http_post_json(
        "https://api.groq.com/openai/v1/chat/completions",
        {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        payload,
    )
    choice = ((data.get("choices") or [{}])[0]).get("message", {})
    text = choice.get("content", "")
    usage = data.get("usage") or {}
    return {
        "text": (text or "").strip(),
        "usage": {
            "input_tokens": usage.get("prompt_tokens", rough_tokens(system + prompt)),
            "output_tokens": usage.get("completion_tokens", rough_tokens(text)),
        },
    }


def call_gemini(api_key, model, system, prompt, max_tokens, temperature):
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
    full_prompt = f"SYSTEM:\n{system}\n\nUSER:\n{prompt}"
    payload = {
        "contents": [{"role": "user", "parts": [{"text": full_prompt}]}],
        "generationConfig": {
            "temperature": temperature,
            "maxOutputTokens": max_tokens,
        },
    }
    data = http_post_json(url, {"Content-Type": "application/json"}, payload)
    candidates = data.get("candidates") or []
    text_parts = []
    if candidates:
        parts = (((candidates[0].get("content") or {}).get("parts")) or [])
        for part in parts:
            if "text" in part:
                text_parts.append(part["text"])
    text = "\n".join(text_parts).strip()
    usage = data.get("usageMetadata") or {}
    return {
        "text": text,
        "usage": {
            "input_tokens": usage.get("promptTokenCount", rough_tokens(system + prompt)),
            "output_tokens": usage.get("candidatesTokenCount", rough_tokens(text)),
        },
    }


def call_provider(provider, api_keys, settings, system, prompt, max_tokens):
    model = settings["models"].get(provider, "")
    temperature = settings.get("temperature", 0.2)
    if provider == "claude":
        result = call_claude(api_keys["claude"], model, system, prompt, max_tokens, temperature)
    elif provider == "groq":
        result = call_groq(api_keys["groq"], model, system, prompt, max_tokens, temperature)
    elif provider == "gemini":
        result = call_gemini(api_keys["gemini"], model, system, prompt, max_tokens, temperature)
    else:
        raise RuntimeError(f"Unknown provider: {provider}")
    result["provider"] = provider
    result["model"] = model
    return result


def call_with_fallback(preferred_providers, api_keys, settings, system, prompt, max_tokens):
    errors = []
    for provider in preferred_providers:
        if not provider_available(api_keys, provider):
            continue
        try:
            return call_provider(provider, api_keys, settings, system, prompt, max_tokens)
        except Exception as e:
            errors.append(f"{provider}: {e}")
    if errors:
        raise RuntimeError(" | ".join(errors))
    raise RuntimeError("No provider available for this step")


def session_document_list(session):
    docs = sorted((session.get("documents") or {}).values(), key=lambda d: d.get("added_at", 0), reverse=True)
    return [
        {
            "id": doc.get("id"),
            "name": doc.get("name"),
            "size_chars": doc.get("size_chars", 0),
            "chunks": len(doc.get("chunks") or []),
        }
        for doc in docs
    ]


def orchestrate(message, session, settings, api_keys, attachments=None):
    context_pack = build_context_pack(session)
    route = choose_route(message, settings.get("mode", "auto"))
    trace = []
    usage = usage_accumulator()

    added_documents = ingest_attachments(session, attachments or [], settings)
    doc_grounding = build_document_grounding(message, session, settings)
    web_grounding = build_web_grounding(message, settings)
    overview = grounding_overview(doc_grounding, web_grounding)
    grounded_text = build_grounding_text(doc_grounding, web_grounding)

    if added_documents:
        trace.append({
            "step": "Files added",
            "provider": "local",
            "model": "session-docs",
            "output": "\n".join(f"{item['name']} · {item['chunks']} chunks · {item['size_chars']} chars" for item in added_documents),
            "usage": {"input_tokens": 0, "output_tokens": 0},
        })

    if doc_grounding.get("count"):
        trace.append({
            "step": "Document retrieval",
            "provider": "local",
            "model": "keyword-chunks",
            "output": doc_grounding.get("context", ""),
            "usage": {"input_tokens": 0, "output_tokens": 0},
        })

    if web_grounding.get("enabled"):
        web_output = web_grounding.get("context") or "No web results found."
        if web_grounding.get("errors"):
            web_output += "\n\nSearch notes:\n" + "\n".join(web_grounding["errors"][:4])
        trace.append({
            "step": "Web search",
            "provider": "web",
            "model": settings.get("web_search", {}).get("engine", "search"),
            "output": web_output,
            "usage": {"input_tokens": 0, "output_tokens": 0},
        })

    def run_step(label, preferred_providers, system, prompt, budget):
        result = call_with_fallback(preferred_providers, api_keys, settings, system, prompt, budget)
        trace.append({
            "step": label,
            "provider": result["provider"],
            "model": result["model"],
            "output": result["text"],
            "usage": result["usage"],
        })
        add_usage(usage, result["provider"], result["usage"])
        return result["text"]

    if route == "groq_only":
        final = run_step(
            "Direct answer",
            ["groq", "gemini", "claude"],
            BUILDER_SYSTEM,
            (
                f"{context_pack}\n\n{overview}\n\nLatest user request:\n{message}\n\n"
                f"Grounded context:\n{grounded_text or 'None'}\n\n"
                "Instructions:\n"
                "- If web sources are provided, cite them inline like [1], [2].\n"
                "- If document snippets are provided, cite them like [D1], [D2].\n"
                "- Do not invent citations.\n\n"
                "Write the best final answer."
            ).strip(),
            settings["budgets"]["builder"],
        )
        return {
            "route": route,
            "final": final,
            "trace": trace,
            "usage": usage,
            "added_documents": added_documents,
            "documents": session_document_list(session),
            "web_search": web_grounding,
        }

    plan_text = run_step(
        "Plan",
        ["claude", "gemini", "groq"],
        PLANNER_SYSTEM,
        make_plan_prompt(message, context_pack, overview),
        settings["budgets"]["planner"],
    )

    research_text = run_step(
        "Research / critique",
        ["groq", "gemini", "claude"],
        RESEARCH_SYSTEM,
        make_research_prompt(message, context_pack, plan_text, grounded_text),
        settings["budgets"]["research"],
    )

    draft_text = run_step(
        "Build",
        ["gemini", "claude", "groq"],
        BUILDER_SYSTEM,
        make_builder_prompt(message, context_pack, plan_text, research_text, grounded_text),
        settings["budgets"]["builder"],
    )

    final_text = draft_text
    if route == "collective":
        final_text = run_step(
            "Verify",
            ["claude", "gemini", "groq"],
            VERIFIER_SYSTEM,
            make_verifier_prompt(message, context_pack, plan_text, research_text, draft_text, grounded_text),
            settings["budgets"]["verifier"],
        )

    return {
        "route": route,
        "final": final_text,
        "trace": trace,
        "usage": usage,
        "added_documents": added_documents,
        "documents": session_document_list(session),
        "web_search": web_grounding,
    }


class Handler(BaseHTTPRequestHandler):
    def _send(self, code, payload, content_type="application/json; charset=utf-8"):
        raw = payload if isinstance(payload, (bytes, bytearray)) else payload.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(raw)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(raw)

    def _json(self, code, obj):
        self._send(code, json.dumps(obj).encode("utf-8"))

    def log_message(self, fmt, *args):
        return

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == "/" or parsed.path == "/index.html":
            if not os.path.exists(INDEX_HTML_PATH):
                self._send(500, b"Missing index.html", "text/plain; charset=utf-8")
                return
            with open(INDEX_HTML_PATH, "rb") as f:
                self._send(200, f.read(), "text/html; charset=utf-8")
            return
        if parsed.path == "/api/health":
            self._json(200, {"ok": True, "time": now_ts()})
            return
        self._send(404, b"Not found", "text/plain; charset=utf-8")

    def do_POST(self):
        parsed = urlparse(self.path)
        if parsed.path != "/api/chat":
            self._send(404, b"Not found", "text/plain; charset=utf-8")
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(length).decode("utf-8")
            payload = json.loads(raw or "{}")
        except Exception:
            self._json(400, {"error": "Invalid JSON body"})
            return

        message = (payload.get("message") or "").strip()
        if not message:
            self._json(400, {"error": "Message is required"})
            return

        settings = deep_merge(DEFAULT_SETTINGS, payload.get("settings") or {})
        api_keys = get_api_keys(payload)
        if not any(provider_available(api_keys, p) for p in ["claude", "groq", "gemini"]):
            self._json(400, {"error": "Add at least one API key (Claude, Groq, or Gemini)."})
            return

        attachments = payload.get("attachments") or []
        session_id = (payload.get("sessionId") or str(uuid.uuid4())).strip()
        session = SESSIONS.setdefault(session_id, {"summary": "", "turns": [], "documents": {}, "created_at": now_ts()})

        try:
            result = orchestrate(message, session, settings, api_keys, attachments=attachments)
            session["turns"].append({"role": "user", "content": message})
            session["turns"].append({"role": "assistant", "content": result["final"]})
            summarize_session_if_needed(session, api_keys, settings)
            cost = estimate_cost(result["usage"], settings)
        except Exception as e:
            self._json(500, {"error": str(e)})
            return

        self._json(200, {
            "sessionId": session_id,
            "route": result["route"],
            "final": result["final"],
            "trace": result["trace"],
            "usage": result["usage"],
            "cost": cost,
            "memorySummary": session.get("summary", ""),
            "addedDocuments": result.get("added_documents", []),
            "documents": result.get("documents", []),
            "webSearch": result.get("web_search", {}),
        })


def main():
    server = HTTPServer((HOST, PORT), Handler)
    print(f"Tri-model chat running on http://localhost:{PORT}")
    server.serve_forever()


if __name__ == "__main__":
    main()
