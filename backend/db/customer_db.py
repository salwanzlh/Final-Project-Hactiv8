import json
from pathlib import Path
from functools import lru_cache

_DATA_PATH = Path(__file__).parent / "customer_history.json"


@lru_cache(maxsize=1)
def get_all_customers() -> list[dict]:
    with open(_DATA_PATH, encoding="utf-8") as f:
        return json.load(f)
