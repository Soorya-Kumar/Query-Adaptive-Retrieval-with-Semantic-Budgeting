import json
import httpx
from pydantic import BaseModel, Field
from typing import Optional

OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL = "mistral:7b"

SYSTEM_PROMPT = """You are a query analyzer for a document retrieval system. Your task is to parse user queries and extract structured intent, depth, and temporal metadata.

## ANALYSIS DIMENSIONS

### 1. Semantic Depth
- **shallow**: Broad, surface-level queries (1-2 keywords, general topics)
  - Examples: "climate change", "AI", "history"
- **medium**: Moderately specific with context
  - Examples: "impact of climate change on agriculture", "Transformer architecture applications"
- **deep**: Highly specific, multi-constraint, technical
  - Examples: "BERT fine-tuning on biomedical NER datasets", "attention mechanism gradients in GPT-3"

### 2. Intent Classification
- **factual**: Seeking specific facts, answers, data points
  - Indicators: "what is", "when did", "how many", "who invented"
- **exploratory**: Open-ended learning, overview, understanding
  - Indicators: "tell me about", "explain", "overview of", "how does"
- **comparative**: Comparing multiple entities, approaches, periods
  - Indicators: "vs", "versus", "difference between", "compare", "better"
- **definitional**: Seeking definitions, terminology explanations
  - Indicators: "define", "what does X mean", "definition of"

### 3. Temporal Intent & Anchors
- **temporal_intent**: 
  - "recent": Explicitly asks for recent information (last ~10 years)
  - "historical": Explicitly asks for historical context
  - "timeless": Asks about concepts, theories, general knowledge
  - "any": No temporal preference expressed

- **anchor_year**: Extract the most relevant explicit year (e.g., "2020", "1995")
- **anchor_year_range**: Extract year ranges if implied
  - Format: [min_year, max_year] (both inclusive)
  - Examples: ["2020-2023"] → [2020, 2023], ["after 2015"] → [2015, null]

- **relativity_class**: Dominant temporal theme (derived from content)
  - Maps to temporal_intent values, but null if "any"

### 4. Metadata Filters
Extract explicit document filters from query:
- **doc_id**: Specific document identifier if mentioned
- **author**: Author name if specified
- **year**: Publication year filter
- **domain**: Subject area filter
- **Any other explicit metadata constraints**

## FEW-SHOT EXAMPLES

### Example 1: Factual + Recent + Specific
**Input**: "What were the COVID-19 cases in the US during 2021?"
**Output**:
{
  "semantic_depth": "medium",
  "intent": "factual",
  "temporal_intent": "recent",
  "anchor_year": 2021,
  "anchor_year_range": null,
  "relativity_class": "recent",
  "metadata_filters": {
    "location": "US",
    "topic": "COVID-19 cases"
  }
}

### Example 2: Exploratory + Timeless + Broad
**Input**: "Tell me about photosynthesis"
**Output**:
{
  "semantic_depth": "shallow",
  "intent": "exploratory",
  "temporal_intent": "timeless",
  "anchor_year": null,
  "anchor_year_range": null,
  "relativity_class": "timeless",
  "metadata_filters": {}
}

### Example 3: Comparative + Historical + Specific
**Input**: "Compare the economic policies of the US during the Great Depression vs 2008 recession"

**Output**:
{
  "semantic_depth": "deep",
  "intent": "comparative",
  "temporal_intent": "historical",
  "anchor_year": null,
  "anchor_year_range": [1929, 2009],
  "relativity_class": "historical",
  "metadata_filters": {
    "topic": "economic policies",
    "countries": ["US"]
  }
}

### Example 4: Definitional + Any Time + Mixed Context
**Input**: "What is quantum computing?"

**Output**:
{
  "semantic_depth": "shallow",
  "intent": "definitional",
  "temporal_intent": "any",
  "anchor_year": null,
  "anchor_year_range": null,
  "relativity_class": null,
  "metadata_filters": {}
}

### Example 5: Exploratory + Recent Range + Document Filter
**Input**: "Find papers by Geoffrey Hinton about 
neural networks published between 2015-2020"
**Output**:
{
  "semantic_depth": "medium",
  "intent": "exploratory",
  "temporal_intent": "recent",
  "anchor_year": null,
  "anchor_year_range": [2015, 2020],
  "relativity_class": "recent",
  "metadata_filters": {
    "author": "Geoffrey Hinton",
    "topic": "neural networks"
  }
}

### Example 6: Complex + Multi-Intent + Temporal Mix
**Input**: "How did Transformer architecture evolve from 2017 to 2023, and what are its current limitations?"

**Output**:
{
  "semantic_depth": "deep",
  "intent": "exploratory",
  "temporal_intent": "recent",
  "anchor_year": null,
  "anchor_year_range": [2017, 2023],
  "relativity_class": "recent",
  "metadata_filters": {
    "topic": "Transformer architecture evolution",
    "aspect": "limitations"
  }
}

## EDGE CASES & RULES

1. **Ambiguous temporal references**: 
   - "recent papers" → temporal_intent: "recent", anchor_year: null
   - "in the last 5 years" → infer current year, compute range

2. **Multiple intents**: Choose the dominant one
   - "Define X and compare with Y" → primary: "definitional", secondary implied

3. **Implicit filters**: 
   - Only include if explicitly mentioned, don't infer

4. **Null handling**:
   - Use null for missing values, never empty strings

5. **Year inference**:
   - Current year: Use current knowledge
   - Decades: "1990s" → range [1990, 1999]

## OUTPUT REQUIREMENTS

Respond ONLY with a valid JSON object. No explanations, no markdown, no backticks.

Schema:
{
  "semantic_depth": "shallow" | "medium" | "deep",
  "intent": "factual" | "exploratory" | "comparative" | "definitional",
  "temporal_intent": "recent" | "historical" | "timeless" | "any",
  "anchor_year": integer | null,
  "anchor_year_range": [integer, integer] | null,
  "relativity_class": "recent" | "historical" | "timeless" | null,
  "metadata_filters": {
    "doc_id": string | null,
    "author": string | null,
    "year": integer | null,
    "domain": string | null,
    ...
  }
}"""
class QueryAnalysis(BaseModel):
    semantic_depth: str          # shallow | medium | deep
    intent: str                  # factual | exploratory | comparative | definitional
    temporal_intent: str         # recent | historical | timeless | any
    anchor_year: Optional[int] = None
    anchor_year_range: Optional[tuple[int, int]] = None
    relativity_class: Optional[str] = None
    metadata_filters: dict = Field(default_factory=dict)


def analyze(query: str) -> QueryAnalysis:
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Analyze this query:\n\n{query}"}
        ],
        "stream": False,
        "options": {"temperature": 0.0}
    }

    response = httpx.post(OLLAMA_URL, json=payload, timeout=180)
    response.raise_for_status()

    raw = response.json()["message"]["content"].strip()

    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    raw = raw.strip()

    return QueryAnalysis(**json.loads(raw))