#!/usr/bin/env python3
"""kernel/memory_store.py — the kernel's small, persistent, HARD-BOUNDED memory store (default cap 100 MB).

Design goals (user): a small persistent memory that is ALWAYS consulted, managed well within HARD boundaries, and
kept up to date — governed by a self-describing 'memory store object' that tells the kernel to do so. Reached
THROUGH THE SCHEDULER (registered as the capability-gated syscall-tier op:memory).

  * Persistent + durable: SQLite (WAL), one file.
  * HARD cap: every put accounts bytes; if it wouldn't fit, evict least-recently-used NON-PINNED entries to make
    room; if it STILL can't fit (entry too big, or only pinned remain) the put is REJECTED — the cap is never
    exceeded.
  * GOVERNANCE OBJECT: a pinned, never-evicted entry (__governance__) whose content is the consult/maintain policy;
    consult() always returns it first. It is the 'object telling it to do so'.
  * Kept up to date: put/update/delete + prune(); the governance policy instructs the kernel to maintain it.
"""
from __future__ import annotations
import os, json, time, sqlite3, threading

GOV_KEY = "__governance__"
DEFAULT_MAX_BYTES = 100 * 1024 * 1024            # 100 MB hard cap

GOVERNANCE_POLICY = (
    "KERNEL MEMORY — POLICY (this object is pinned and always returned first by consult()). "
    "1) ALWAYS CONSULT this store at the start of a task: check for relevant durable facts, proven helpers, and "
    "preferences BEFORE synthesizing or asking. "
    "2) KEEP IT CURRENT: when you learn a durable fact or prove a helper, PUT it; UPDATE entries that change; "
    "DELETE entries that become wrong. Prefer few high-value entries over many. "
    "3) STAY WITHIN BOUNDS: the store is HARD-CAPPED (see stats.max_bytes); when near the cap the least-recently-"
    "used non-pinned entries are evicted automatically, and an oversized write is REJECTED. Record only DURABLE, "
    "high-value items (facts, proven primitives, preferences) — never transient run state. "
    "4) This governance object is pinned: it is never evicted and should be kept accurate."
)


class MemoryStore:
    def __init__(self, path: str, max_bytes: int = DEFAULT_MAX_BYTES):
        self.path = path
        self.max_bytes = int(max_bytes)
        # One sqlite connection is shared with check_same_thread=False so the scheduler can call op:memory from any
        # thread; concurrent use of a shared connection corrupts SQLite internals (observed: SIGSEGV in
        # sqlite3Prepare). A reentrant lock serializes ALL db access, which makes the shared connection safe.
        self._lock = threading.RLock()
        gov_b = self._bytes(GOV_KEY, GOVERNANCE_POLICY)      # the store must be able to hold its own policy
        if self.max_bytes < gov_b:
            raise ValueError(f"max_bytes ({self.max_bytes}) too small for the governance object ({gov_b} B)")
        d = os.path.dirname(os.path.abspath(path))
        if d:
            os.makedirs(d, exist_ok=True)
        with self._lock:
            self.db = sqlite3.connect(path, check_same_thread=False)
            self.db.execute("PRAGMA journal_mode=WAL")
            self.db.execute(
                "CREATE TABLE IF NOT EXISTS mem (key TEXT PRIMARY KEY, value TEXT, kind TEXT, pinned INTEGER DEFAULT 0,"
                " created REAL, updated REAL, accessed REAL, n_access INTEGER DEFAULT 0, bytes INTEGER)")
            self.db.commit()
        self._ensure_governance()

    # ── helpers ──
    @staticmethod
    def _now():
        return time.time()

    @staticmethod
    def _bytes(key: str, value: str) -> int:
        return len(key.encode("utf-8")) + len(value.encode("utf-8"))

    def total_bytes(self) -> int:
        with self._lock:
            return self.db.execute("SELECT COALESCE(SUM(bytes), 0) FROM mem").fetchone()[0]

    def _ensure_governance(self):
        with self._lock:
            if self.db.execute("SELECT 1 FROM mem WHERE key=?", (GOV_KEY,)).fetchone() is None:
                self.put(GOV_KEY, GOVERNANCE_POLICY, kind="governance", pin=True)

    def _evict_to_fit(self, need: int, protect: str) -> int:
        """Evict least-recently-used NON-PINNED (and not `protect`) entries until `need` more bytes fit. Returns
        how many were evicted. Never touches pinned entries (incl. governance). Caller holds self._lock."""
        with self._lock:
            evicted = 0
            while self.total_bytes() + need > self.max_bytes:
                row = self.db.execute(
                    "SELECT key FROM mem WHERE pinned=0 AND key!=? ORDER BY accessed ASC, updated ASC LIMIT 1",
                    (protect,)).fetchone()
                if row is None:
                    break                                # only pinned/protected remain — caller will reject
                self.db.execute("DELETE FROM mem WHERE key=?", (row[0],))
                evicted += 1
            return evicted

    # ── API ──
    def put(self, key: str, value, kind: str = "note", pin: bool = False) -> dict:
        if not key or key.startswith("__") and key != GOV_KEY:
            return {"ok": False, "error": "keys starting with __ are reserved"}
        try:
            v = value if isinstance(value, str) else json.dumps(value, default=str, sort_keys=True)
        except (TypeError, ValueError) as e:
            return {"ok": False, "error": f"value not serializable: {e}"}
        b = self._bytes(key, v)
        if b > self.max_bytes:
            return {"ok": False, "error": f"entry ({b} B) exceeds store cap ({self.max_bytes} B)"}
        with self._lock:
            if key == GOV_KEY and self.db.execute("SELECT 1 FROM mem WHERE key=?", (GOV_KEY,)).fetchone() is not None:
                return {"ok": False, "error": "governance object is immutable via put()"}   # no policy self-tamper
            existing = self.db.execute("SELECT bytes, pinned FROM mem WHERE key=?", (key,)).fetchone()
            delta = b - (existing[0] if existing else 0)
            evicted = self._evict_to_fit(delta, protect=key) if delta > 0 else 0
            if self.total_bytes() + delta > self.max_bytes:  # still doesn't fit (only pinned left) -> REJECT
                return {"ok": False, "error": "store full of pinned entries; cannot fit (hard cap)", "evicted": evicted}
            now = self._now()
            pinned = 1 if (pin or (existing and existing[1])) else 0
            self.db.execute(
                "INSERT INTO mem(key,value,kind,pinned,created,updated,accessed,n_access,bytes) "
                "VALUES(?,?,?,?,?,?,?,0,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value, kind=excluded.kind, "
                "pinned=excluded.pinned, updated=excluded.updated, accessed=excluded.updated, bytes=excluded.bytes",
                (key, v, kind, pinned, now, now, now, b))
            self.db.commit()
            return {"ok": True, "key": key, "bytes": b, "evicted": evicted, "total_bytes": self.total_bytes()}

    def get(self, key: str, touch: bool = True):
        with self._lock:
            row = self.db.execute("SELECT value FROM mem WHERE key=?", (key,)).fetchone()
            if row is None:
                return None
            if touch:
                self.db.execute("UPDATE mem SET accessed=?, n_access=n_access+1 WHERE key=?", (self._now(), key))
                self.db.commit()
            return row[0]

    def delete(self, key: str) -> dict:
        if key == GOV_KEY:
            return {"ok": False, "error": "the governance object cannot be deleted"}
        with self._lock:
            cur = self.db.execute("DELETE FROM mem WHERE key=?", (key,))
            self.db.commit()
            return {"ok": cur.rowcount > 0, "deleted": cur.rowcount}

    def list(self, limit: int = 100) -> list:
        with self._lock:
            rows = self.db.execute(
                "SELECT key, kind, pinned, bytes, updated, n_access FROM mem ORDER BY pinned DESC, updated DESC LIMIT ?",
                (limit,)).fetchall()
        return [{"key": k, "kind": ki, "pinned": bool(p), "bytes": b, "updated": u, "n_access": na}
                for k, ki, p, b, u, na in rows]

    def stats(self) -> dict:
        with self._lock:
            n = self.db.execute("SELECT COUNT(*) FROM mem").fetchone()[0]
            tot = self.total_bytes()
        return {"entries": n, "total_bytes": tot, "max_bytes": self.max_bytes,
                "pct_full": round(100.0 * tot / self.max_bytes, 3), "path": self.path}

    def consult(self, limit: int = 10) -> dict:
        """The ALWAYS-CONSULT entry point: returns the governance policy FIRST, then a digest of recent entries +
        store stats. The kernel calls this before acting; the governance object tells it to."""
        with self._lock:
            recent = self.db.execute(
                "SELECT key, kind, updated FROM mem WHERE key!=? ORDER BY updated DESC LIMIT ?",
                (GOV_KEY, limit)).fetchall()
            return {"governance": self.get(GOV_KEY, touch=False),
                    "recent": [{"key": k, "kind": ki, "updated": u} for k, ki, u in recent],
                    "stats": self.stats()}

    def close(self):
        try:
            self.db.close()
        except Exception:
            pass


# ── scheduler integration: op:memory (action-dispatched, capability-gated, syscall tier) ──
def memory_handler_for(store: "MemoryStore"):
    """Build a Scheduler executor-handler bound to `store`; dispatches inputs['action']."""
    def handler(sched, inputs):
        inputs = inputs or {}
        action = inputs.get("action", "consult")
        if action == "consult":
            return store.consult(int(inputs.get("limit", 10)))
        if action == "get":
            v = store.get(inputs.get("key", ""))
            return {"key": inputs.get("key"), "value": v, "found": v is not None}
        if action == "put":
            return store.put(inputs.get("key", ""), inputs.get("value"), inputs.get("kind", "note"),
                             bool(inputs.get("pin", False)))
        if action == "list":
            return {"entries": store.list(int(inputs.get("limit", 100)))}
        if action == "delete":
            return store.delete(inputs.get("key", ""))
        if action == "stats":
            return store.stats()
        return {"error": f"unknown memory action {action!r}"}
    return handler


def register_memory_tool(sched, store: "MemoryStore", grant: bool = True) -> dict:
    """Register the persistent memory as the capability-gated syscall-tier tool op:memory."""
    try:
        from .tools import ToolSpec, register_tool
    except ImportError:
        from tools import ToolSpec, register_tool
    spec = ToolSpec(name="memory", handler=memory_handler_for(store),
                    description="The kernel's PERSISTENT, hard-bounded memory (default 100MB) — ALWAYS consult it "
                                "first. actions: consult (governance policy + recent digest + stats), get(key), "
                                "put(key,value,kind,pin), list, delete(key), stats. Record durable facts + proven "
                                "helpers; it self-evicts least-recently-used non-pinned entries within the cap.",
                    signature="memory(action='consult'|'get'|'put'|'list'|'delete'|'stats', key?, value?, kind?, pin?, limit?)")
    return register_tool(sched, spec, grant=grant)


__all__ = ["MemoryStore", "GOV_KEY", "GOVERNANCE_POLICY", "DEFAULT_MAX_BYTES",
           "memory_handler_for", "register_memory_tool"]
