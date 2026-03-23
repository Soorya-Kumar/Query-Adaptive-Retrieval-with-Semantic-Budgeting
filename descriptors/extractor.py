import re
import json
from descriptors.schema import ChunkDescriptor
from utils import post_with_retries

OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL = "mistral:7b"

SYSTEM_PROMPT = """
You are a semantic indexing engine for a multi-stage document retrieval system.

Extract hierarchical semantic tags optimized for retrieval, clustering, and reranking.

-----------------------------------
TAG LEVELS (STRICT SEMANTIC SEPARATION)
-----------------------------------
fine (4–8):
- Specific entities: models, algorithms, datasets, metrics, chemicals, methods
- Must be concrete and discriminative
- Examples: "BERT", "ResNet", "cross-entropy loss", "chlorophyll"

mid (3–5):
- Topics or methodological categories grouping fine tags
- Examples: "transformer architectures", "supervised learning", "plant physiology"

coarse (2–3):
- Broad domains using standardized academic terms
- Examples: "Machine Learning", "Biology", "Computer Vision"

-----------------------------------
CRITICAL CONSTRAINTS
-----------------------------------
- NO redundancy across levels
- Ignore the random noise (numbers following .X) at the end of the chunk
- NO vague terms (e.g., "system", "approach", "study")
- Tags must be 1–3 words (prefer canonical forms)
- Prefer terms used in research literature
- Avoid overly generic tags unless central
- These will be embedded so ensure semantic diversity within each level

-----------------------------------
CONFIDENCE SCORING
-----------------------------------
1.0 → explicitly central
0.8–0.95 → clearly present
0.6–0.75 → implied
<0.6 → EXCLUDE

-----------------------------------
TEMPORAL METADATA
-----------------------------------
anchor_year:
- Extract if explicitly mentioned
- Else null

relativity_class:
- "recent" → modern (post-2015 methods, deep learning era)
- "historical" → foundational (pre-2000)
- "timeless" → scientific facts or general principles

-----------------------------------
FEW-SHOT EXAMPLES (HIGH QUALITY)
-----------------------------------

Input:
"Transformer models like BERT and GPT use attention mechanisms to improve NLP tasks such as question answering and text classification."

Output:
{
  "fine": [
    {"tag": "Transformer", "confidence": 1.0},
    {"tag": "BERT", "confidence": 0.95},
    {"tag": "GPT", "confidence": 0.9},
    {"tag": "attention mechanism", "confidence": 0.95},
    {"tag": "question answering", "confidence": 0.85},
    {"tag": "text classification", "confidence": 0.85}
  ],
  "mid": [
    {"tag": "transformer architectures", "confidence": 0.95},
    {"tag": "natural language tasks", "confidence": 0.9}
  ],
  "coarse": [
    {"tag": "Natural Language Processing", "confidence": 1.0},
    {"tag": "Machine Learning", "confidence": 0.9}
  ],
  "anchor_year": null,
  "relativity_class": "recent"
}

Input:
"Photosynthesis in plants converts light energy into chemical energy using chlorophyll in chloroplasts."

Output:
{
  "fine": [
    {"tag": "photosynthesis", "confidence": 1.0},
    {"tag": "chlorophyll", "confidence": 0.95},
    {"tag": "chloroplast", "confidence": 0.9},
    {"tag": "light energy", "confidence": 0.85}
  ],
  "mid": [
    {"tag": "plant physiology", "confidence": 0.95},
    {"tag": "energy conversion", "confidence": 0.85}
  ],
  "coarse": [
    {"tag": "Biology", "confidence": 1.0}
  ],
  "anchor_year": null,
  "relativity_class": "timeless"
}

-----------------------------------
OUTPUT FORMAT (STRICT JSON ONLY)
-----------------------------------
{
  "fine": [{"tag": "...", "confidence": 0.0}],
  "mid": [{"tag": "...", "confidence": 0.0}],
  "coarse": [{"tag": "...", "confidence": 0.0}],
  "anchor_year": null,
  "relativity_class": "timeless"
}

-----------------------------------
OBJECTIVE
-----------------------------------
Maximize usefulness for:
- hybrid retrieval (BM25 + dense)
- embedding-based reranking
- semantic clustering

Prefer tags that improve discrimination across documents.
"""

REQUIRED_KEYS = {"fine", "mid", "coarse", "relativity_class"}
VALID_RELATIVITY = {"recent", "historical", "timeless"}


def _sanitize_candidate(candidate: str) -> str:
    """
    Fix known LLM JSON quirks before passing to json.loads.

    Handles:
    - Unquoted placeholder years:  "anchor_year": 20XX  → "anchor_year": null
    - Partial years / ranges:      "anchor_year": 2020-2023  → "anchor_year": null
    - Trailing commas before } or ]: {"a":1,}  → {"a":1}
    - Single-quoted strings:       {'tag': 'BERT'}  → {"tag": "BERT"}
    """
    # Unquoted anchor_year values that aren't plain integers or null
    # Matches: 20XX, 2020s, 2020-2023, ~2020, "around 2020", etc. when unquoted
    candidate = re.sub(
        r'("anchor_year"\s*:\s*)(?!null|"|\d{4}\b)([^\n,}\]]+)',
        r'\1null',
        candidate,
    )
    # Unquoted year-like values with non-digit characters: 20XX, 2020s, 2020-2023
    candidate = re.sub(
        r'("anchor_year"\s*:\s*)(\d{4}[^\d\s,}\]"\']+)',
        r'\1null',
        candidate,
    )
    # Trailing commas before closing brace/bracket
    candidate = re.sub(r',\s*([}\]])', r'\1', candidate)
    # Single-quoted strings → double-quoted (naïve but handles most LLM output)
    # Only replace when single quotes wrap a value (not apostrophes mid-word)
    candidate = re.sub(r"(?<![\\])'([^']*)'", r'"\1"', candidate)
    return candidate


def _extract_json_candidate(raw: str) -> str:
    """
    Try to extract a JSON object from raw LLM output.
    Attempts, in order:
      1. Fenced code block  ```json ... ``` or ``` ... ```
      2. Locate the first '{' and find its matching closing '}'
         by tracking brace depth — handles leading/trailing prose.
    """
    raw = raw.strip()

    # 1. Fenced block
    m = re.search(r"```(?:json)?\s*(.*?)```", raw, re.DOTALL | re.IGNORECASE)
    if m:
        return m.group(1).strip()

    # 2. Brace-depth scan — more reliable than a greedy .*
    start = raw.find("{")
    if start == -1:
        return ""
    depth = 0
    in_string = False
    escape_next = False
    for i, ch in enumerate(raw[start:], start):
        if escape_next:
            escape_next = False
            continue
        if ch == "\\" and in_string:
            escape_next = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return raw[start : i + 1]
    return ""


def _validate_structure(data: dict) -> None:
    """Raise ValueError with a descriptive message if required fields are missing or malformed."""
    missing = REQUIRED_KEYS - data.keys()
    if missing:
        raise ValueError(f"Parsed JSON is missing required keys: {missing}")

    for level in ("fine", "mid", "coarse"):
        items = data.get(level)
        if not isinstance(items, list):
            raise ValueError(f"'{level}' must be a list, got {type(items).__name__}")
        for item in items:
            if not isinstance(item, dict) or "tag" not in item or "confidence" not in item:
                raise ValueError(
                    f"Each entry in '{level}' must have 'tag' and 'confidence'; got: {item!r}"
                )

    rc = data.get("relativity_class")
    if rc not in VALID_RELATIVITY:
        raise ValueError(
            f"'relativity_class' must be one of {VALID_RELATIVITY}, got {rc!r}"
        )


def _to_int_year(val) -> int | None:
    """Coerce anchor_year to Optional[int], accepting ints, strings, dicts, lists."""
    if val is None:
        return None
    if isinstance(val, int):
        return val
    if isinstance(val, str):
        m = re.search(r"\b(1[0-9]{3}|20[0-9]{2})\b", val)
        return int(m.group(0)) if m else None
    if isinstance(val, dict):
        for k in ("year", "year_of_publication", "date"):
            if k in val:
                result = _to_int_year(val[k])
                if result is not None:
                    return result
        for v in val.values():
            result = _to_int_year(v)
            if result is not None:
                return result
        return None
    if isinstance(val, list):
        for item in val:
            result = _to_int_year(item)
            if result is not None:
                return result
        return None
    return None


def extract_descriptors(chunk_text: str) -> ChunkDescriptor:
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Extract descriptors for this chunk:\n\n{chunk_text}"},
        ],
        "stream": False,
        "options": {"temperature": 0.1},
    }

    response = post_with_retries(OLLAMA_URL, json=payload)
    response.raise_for_status()

    raw = response.json().get("message", {}).get("content", "")
    candidate = _extract_json_candidate(raw)

    if not candidate:
        raise ValueError(
            f"No JSON object found in model response. Full response:\n{raw[:2000]!r}"
        )

    candidate = _sanitize_candidate(candidate)

    try:
        data = json.loads(candidate)
    except json.JSONDecodeError as e:
        raise ValueError(
            f"JSON parse error: {e}\nExtracted candidate ({len(candidate)} chars):\n{candidate[:1000]!r}"
        ) from e

    if not isinstance(data, dict):
        raise ValueError(f"Expected a JSON object, got {type(data).__name__}: {candidate[:200]!r}")

    _validate_structure(data)

    data["anchor_year"] = _to_int_year(data.get("anchor_year"))
    return ChunkDescriptor(**data)