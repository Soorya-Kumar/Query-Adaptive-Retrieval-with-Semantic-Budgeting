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


def extract_descriptors(chunk_text: str) -> ChunkDescriptor:
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Extract descriptors for this chunk:\n\n{chunk_text}"}
        ],
        "stream": False,
        "options": {"temperature": 0.1}
    }

    response = post_with_retries(OLLAMA_URL, json=payload)
    response.raise_for_status()

    raw = response.json()["message"]["content"].strip()

    # Strip accidental markdown fences
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    raw = raw.strip()

    data = json.loads(raw)        
    # Normalize anchor_year to Optional[int]
    ay = data.get("anchor_year", None)

    def _to_int_year(val):
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
                    return _to_int_year(val[k])
            # try any numeric-like fields
            for v in val.values():
                y = _to_int_year(v)
                if y:
                    return y
            return None
        if isinstance(val, list):
            # try first plausible element
            for item in val:
                y = _to_int_year(item)
                if y:
                    return y
            return None
        return None

    data["anchor_year"] = _to_int_year(ay)
    return ChunkDescriptor(**data)