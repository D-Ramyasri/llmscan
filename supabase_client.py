import hashlib
import os
from datetime import datetime
from typing import Any, Dict, Optional

try:
    import streamlit as st
except Exception:  # pragma: no cover - allows importing outside Streamlit
    st = None

try:
    from dotenv import load_dotenv
except Exception:  # pragma: no cover
    load_dotenv = None

try:
    from supabase import create_client
except Exception:  # pragma: no cover
    create_client = None


if load_dotenv is not None:
    _HERE = os.path.dirname(__file__)
    load_dotenv(os.path.join(_HERE, ".env"))


DEFAULT_USER_STATE = {
    "history": [],
    "selected_model": "Qwen/Qwen2.5-0.5B-Instruct",
    "selected_dataset": "Custom",
    "layer_mode": "Auto",
    "manual_layer_idx": 0,
    "strategy": "scale",
    "scale_factor": 0.5,
    "prompt": "",
    "scan_results": None,
    "intervene_results": None,
    "active_chat_id": None,
}

_SUPABASE = None
_SUPABASE_ERROR = None
_SUPABASE_STORAGE_DISABLED = False


def hash_password(pw: str) -> str:
    return hashlib.sha256(pw.encode("utf-8")).hexdigest() if pw is not None else ""


def _now_iso() -> str:
    return datetime.utcnow().isoformat()


def _session_state() -> Dict[str, Any]:
    if st is not None:
        st.session_state.setdefault("_session_only_users", {})
        st.session_state.setdefault("_session_only_remembered_user", "")
        return st.session_state
    return {"_session_only_users": {}, "_session_only_remembered_user": ""}


def _normalize_email(email: str) -> str:
    return (email or "").strip().lower()


def _split_password_hash(state: Dict[str, Any]) -> tuple[str, Dict[str, Any]]:
    clean_state = dict(state or {})
    password_hash = clean_state.pop("password_hash", "")
    return password_hash, clean_state


def _merge_user_row(row: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not row:
        return {}
    state = row.get("state") or {}
    if not isinstance(state, dict):
        state = {}
    merged = dict(DEFAULT_USER_STATE)
    merged.update(state)
    merged["password_hash"] = row.get("password_hash", "")
    return merged


def _client():
    global _SUPABASE, _SUPABASE_ERROR

    if _SUPABASE_STORAGE_DISABLED:
        return None

    if _SUPABASE is not None:
        return _SUPABASE

    url = os.getenv("SUPABASE_URL", "").strip()
    key = os.getenv("SUPABASE_KEY", "").strip()
    if not url or not key:
        _SUPABASE_ERROR = "SUPABASE_URL and SUPABASE_KEY are not configured."
        return None
    if create_client is None:
        _SUPABASE_ERROR = "The supabase package is not installed."
        return None

    try:
        _SUPABASE = create_client(url, key)
        _SUPABASE_ERROR = None
        return _SUPABASE
    except Exception as exc:
        _SUPABASE_ERROR = f"Could not initialize Supabase: {exc}"
        return None


def is_supabase_configured() -> bool:
    return _client() is not None


def get_storage_error() -> Optional[str]:
    _client()
    return _SUPABASE_ERROR


def _is_missing_users_table_error(exc: Exception) -> bool:
    text = str(exc)
    return "PGRST205" in text or "Could not find the table 'public.users'" in text


def _disable_supabase_storage(message: str) -> None:
    global _SUPABASE, _SUPABASE_ERROR, _SUPABASE_STORAGE_DISABLED
    _SUPABASE = None
    _SUPABASE_STORAGE_DISABLED = True
    _SUPABASE_ERROR = message
    _session_state()["_supabase_last_error"] = message


def _save_session_only_user(email: str, state: Dict[str, Any], password_hash: str = "") -> bool:
    users = _session_state().setdefault("_session_only_users", {})
    existing = dict(users.get(email, {}))
    existing.update(state or {})
    if password_hash:
        existing["password_hash"] = password_hash
    users[email] = existing
    return True


def load_user_data(email: str) -> Dict[str, Any]:
    email = _normalize_email(email)
    client = _client()

    if client is None:
        users = _session_state().setdefault("_session_only_users", {})
        return dict(users.get(email, {}))

    try:
        result = client.table("users").select("*").eq("email", email).limit(1).execute()
        rows = result.data or []
        return _merge_user_row(rows[0] if rows else None)
    except Exception as exc:
        if _is_missing_users_table_error(exc):
            _disable_supabase_storage(
                "Supabase connected, but the public.users table does not exist yet. "
                "Using session-only storage until you run supabase_schema.sql."
            )
        else:
            _session_state()["_supabase_last_error"] = f"Could not load user data: {exc}"
        users = _session_state().setdefault("_session_only_users", {})
        return dict(users.get(email, {}))


def save_user_data(email: str, state: Dict[str, Any]) -> bool:
    email = _normalize_email(email)
    password_hash, clean_state = _split_password_hash(state or {})
    client = _client()

    if client is None:
        return _save_session_only_user(email, clean_state, password_hash)

    try:
        existing = client.table("users").select("id,password_hash,state").eq("email", email).limit(1).execute()
        rows = existing.data or []
        if rows:
            current_state = rows[0].get("state") or {}
            if not isinstance(current_state, dict):
                current_state = {}
            current_state.update(clean_state)
            payload = {"state": current_state, "updated_at": _now_iso()}
            if password_hash:
                payload["password_hash"] = password_hash
            client.table("users").update(payload).eq("email", email).execute()
        else:
            client.table("users").insert({
                "email": email,
                "password_hash": password_hash,
                "state": clean_state,
                "created_at": _now_iso(),
                "updated_at": _now_iso(),
            }).execute()
        return True
    except Exception as exc:
        if _is_missing_users_table_error(exc):
            _disable_supabase_storage(
                "Supabase connected, but the public.users table does not exist yet. "
                "Using session-only storage until you run supabase_schema.sql."
            )
        else:
            _session_state()["_supabase_last_error"] = f"Could not save user data: {exc}"
        return _save_session_only_user(email, clean_state, password_hash)


def save_remembered_user(email: str) -> bool:
    # The previous implementation used a local JSON file. With JSON storage removed,
    # this is intentionally session-only unless a browser/app auth layer is added.
    _session_state()["_session_only_remembered_user"] = _normalize_email(email)
    return True


def clear_remembered_user() -> bool:
    _session_state()["_session_only_remembered_user"] = ""
    return True

if __name__ == "__main__":
    print(f"SUPABASE_URL: {os.getenv('SUPABASE_URL', 'NOT FOUND')}")
    print(f"Supabase configured: {is_supabase_configured()}")
