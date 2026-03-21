import json
import httpx
from descriptors.schema import ChunkDescriptor

OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL = "qwen2.5:7b-instruct"

SYSTEM_PROMPT = """Extract semantic descriptors at three levels from text:

fine: specific terms/entities (4-8 tags)
mid: topics/subtopics (3-5 tags)  
coarse: broad domains (2-3 tags)

Confidence: 1.0=core, 0.8=present, 0.6=implied, <0.5=omit

Examples:

Input: "Transformer (2017) revolutionized NLP. BERT (2018) improved on it."
Output: {"fine":[{"tag":"Transformer","confidence":1.0},{"tag":"BERT","confidence":0.95}],"mid":[{"tag":"NLP architectures","confidence":0.9}],"coarse":[{"tag":"NLP","confidence":1.0}],"anchor_year":2017,"relativity_class":"recent"}

Input: "Photosynthesis converts sunlight to energy using chlorophyll."
Output: {"fine":[{"tag":"photosynthesis","confidence":1.0},{"tag":"chlorophyll","confidence":0.95}],"mid":[{"tag":"plant biology","confidence":0.9}],"coarse":[{"tag":"Biology","confidence":1.0}],"anchor_year":null,"relativity_class":"timeless"}

Return ONLY valid JSON. Schema:
{
  "fine": [{"tag": "...", "confidence": 0.0}],
  "mid": [{"tag": "...", "confidence": 0.0}],
  "coarse": [{"tag": "...", "confidence": 0.0}],
  "anchor_year": null,
  "relativity_class": "timeless"
}"""


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

    response = httpx.post(OLLAMA_URL, json=payload, timeout=120)
    response.raise_for_status()

    raw = response.json()["message"]["content"].strip()

    # Strip accidental markdown fences
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    raw = raw.strip()

    data = json.loads(raw)
    return ChunkDescriptor(**data)