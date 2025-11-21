from supabase import create_client, Client
from config.settings import Settings
import os

_client: Client | None = None

def _get() -> Client:
    global _client
    if _client is None:
        s = Settings()
        if not s.supabase_url or not s.supabase_key:
            raise RuntimeError("Supabase not configured")
        _client = create_client(s.supabase_url, s.supabase_key)
    return _client

def sb_upsert(table: str, data: dict):
    return _get().table(table).upsert(data).execute()

def sb_fetch(table: str, limit: int = 50):
    return _get().table(table).select("*").limit(limit).execute()
