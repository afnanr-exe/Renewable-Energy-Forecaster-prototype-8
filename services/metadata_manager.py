import json
import os
from datetime import date

class MetadataManager:
    def __init__(self, path: str):
        self.path = path
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self._data = self._load()

    def _load(self):
        if not os.path.exists(self.path):
            return {}
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}

    def save(self):
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(self._data, f, indent=2)

    def get_last_run(self, key: str):
        return self._data.get(key)

    def set_last_run_today(self, key: str):
        self._data[key] = date.today().isoformat()
        self.save()

    def ran_today(self, key: str) -> bool:
        return self.get_last_run(key) == date.today().isoformat()
