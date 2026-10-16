import hashlib
import os
import pickle
from datetime import datetime, timedelta


class BotCacheManager:
    """Small pickle-backed key/value cache with time-based expiry."""

    def __init__(self, cache_dir="bot_cache", expiry_hours=24):
        self.cache_dir = cache_dir
        self.cache_file = os.path.join(cache_dir, "bot_responses.pkl")
        self.cache_expiry_hours = expiry_hours
        os.makedirs(cache_dir, exist_ok=True)

    def _generate_cache_key(self, key):
        normalized = " ".join(key.lower().split())
        return hashlib.md5(normalized.encode()).hexdigest()

    def _load_cache(self):
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, "rb") as f:
                    return pickle.load(f)
            except (pickle.PickleError, EOFError):
                return {}
        return {}

    def _save_cache(self, cache_data):
        try:
            with open(self.cache_file, "wb") as f:
                pickle.dump(cache_data, f)
        except OSError as e:
            print(f"Error saving cache: {e}")

    def _is_cache_valid(self, cache_entry):
        if not cache_entry or "timestamp" not in cache_entry:
            return False
        cache_time = datetime.fromisoformat(cache_entry["timestamp"])
        return datetime.now() < cache_time + timedelta(hours=self.cache_expiry_hours)

    def get_cached_response(self, key):
        cache_key = self._generate_cache_key(key)
        cache_data = self._load_cache()

        if cache_key in cache_data:
            cache_entry = cache_data[cache_key]
            if self._is_cache_valid(cache_entry):
                return cache_entry["response"]
            del cache_data[cache_key]
            self._save_cache(cache_data)
        return None

    def cache_response(self, key, response_data):
        cache_key = self._generate_cache_key(key)
        cache_data = self._load_cache()
        cache_data[cache_key] = {
            "timestamp": datetime.now().isoformat(),
            "response": response_data,
            "user_input": key,
        }
        self._save_cache(cache_data)

    def clear_cache(self):
        if os.path.exists(self.cache_file):
            os.remove(self.cache_file)

    def get_cache_stats(self):
        cache_data = self._load_cache()
        total_entries = len(cache_data)
        valid_entries = sum(
            1 for entry in cache_data.values() if self._is_cache_valid(entry)
        )
        return {
            "total_entries": total_entries,
            "valid_entries": valid_entries,
            "expired_entries": total_entries - valid_entries,
            "cache_file_size": (
                os.path.getsize(self.cache_file) if os.path.exists(self.cache_file) else 0
            ),
        }
