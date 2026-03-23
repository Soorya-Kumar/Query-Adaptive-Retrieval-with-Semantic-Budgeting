# test_extractor.py
import json
from descriptors import extractor

class MockResponse:
    def __init__(self, content_str):
        self._content = content_str
    def raise_for_status(self):
        return None
    def json(self):
        return {"message": {"content": self._content}}

def run_mock(raw_obj):
    raw_json = json.dumps(raw_obj)
    extractor.post_with_retries = lambda *a, **k: MockResponse(raw_json)
    try:
        desc = extractor.extract_descriptors("dummy chunk text")
        print("Input anchor_year:", raw_obj.get("anchor_year"))
        print("Normalized anchor_year:", desc.anchor_year)
        print("---")
    except Exception as e:
        print("Error:", e)
        print("---")

if __name__ == "__main__":
    cases = [
        # case: list of dicts (like the error)
        {"fine": [], "mid": [], "coarse": [], "anchor_year": [{"year": "1876", "role": "edition"}], "relativity_class": "historical"},
        # case: string
        {"fine": [], "mid": [], "coarse": [], "anchor_year": "1876", "relativity_class": "historical"},
        # case: integer
        {"fine": [], "mid": [], "coarse": [], "anchor_year": 1876, "relativity_class": "historical"},
        # case: dict with date
        {"fine": [], "mid": [], "coarse": [], "anchor_year": {"date": "1876-05-01"}, "relativity_class": "historical"},
        # case: null
        {"fine": [], "mid": [], "coarse": [], "anchor_year": None, "relativity_class": "timeless"},
    ]

    for c in cases:
        run_mock(c)