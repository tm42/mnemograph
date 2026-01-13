# v0.4 Refactor Spec

Fixes identified in code review after SQLite migration.

## 1. Similarity Formula Fix (Critical)

**Bug**: `find_similar()` can return scores > 1.0, causing false duplicate detection.

**Cause**: `substring_score * 0.9 + affix_bonus` = 0.81 + 0.2 = 1.01 for "Concept_1" vs "Concept_10"

**Fix**: Option C - simple max with capped bonus
```python
def _combined_similarity(name1: str, name2: str) -> float:
    base = max(
        _substring_similarity(name1, name2),
        _jaccard_similarity(name1, name2),
        _embedding_similarity(name1, name2) if embeddings else 0.0
    )
    bonus = _affix_bonus(name1, name2)
    return min(base + bonus * 0.1, 1.0)  # Cap at 1.0
```

**File**: `src/mnemograph/engine.py`

---

## 2. SQLite Concurrency (High)

**Issue**: No WAL mode or busy timeout. Concurrent access could fail.

**Fix**: Add to `EventStore._get_conn()`:
```python
conn = sqlite3.connect(str(self.db_path), timeout=30.0)
conn.execute("PRAGMA journal_mode=WAL")
conn.execute("PRAGMA busy_timeout=30000")
```

**File**: `src/mnemograph/events.py`

---

## 3. Compaction Transaction Safety (Critical)

**Issue**: `_compact_events()` does clear + append_batch without transaction wrapper.

**Fix**: Wrap in explicit transaction:
```python
def _compact_events(self):
    events = self.event_store.read_all()
    conn = self.event_store.get_connection()
    try:
        conn.execute("BEGIN IMMEDIATE")
        self.event_store.clear()
        self.event_store.append_batch(events)
        conn.commit()
    except Exception:
        conn.rollback()
        raise
```

**File**: `src/mnemograph/engine.py`

---

## 4. Reload Vector Invalidation (High)

**Issue**: `reload()` rebuilds state but doesn't invalidate vector index.

**Fix**: Add `invalidate()` to VectorIndex and call from reload:
```python
# vectors.py
def invalidate(self):
    """Clear vector index, forcing reindex on next use."""
    conn = self.conn
    if conn:
        conn.execute("DELETE FROM entity_vectors")
        conn.execute("DELETE FROM entity_meta")
        conn.commit()

# engine.py reload()
if self._vector_index:
    self._vector_index.invalidate()
```

**Files**: `src/mnemograph/vectors.py`, `src/mnemograph/engine.py`

---

## 5. clear_graph Confirmation (Medium)

**Issue**: `clear_graph()` deletes everything with no safeguard.

**Fix**: Require reason + confirmation token for large graphs:
```python
def clear_graph(self, reason: str | None = None, confirm_token: str | None = None) -> dict:
    count = len(self.state.entities)
    if count > 10 and confirm_token != f"CLEAR_{count}":
        return {
            "error": "Large graph requires confirmation",
            "entity_count": count,
            "confirm_with": f"CLEAR_{count}"
        }
    # proceed with clear...
```

**File**: `src/mnemograph/engine.py`, update MCP tool schema

---

## 6. Token-Based Observation Limits (Medium)

**Issue**: Unlimited observations per entity can bloat context.

**Fix**: Soft limit with guidance, not hard rejection:
```python
MAX_OBSERVATION_TOKENS = 2000

def add_observations(self, ...):
    current_tokens = sum(len(o.text) // 4 for o in entity.observations)
    new_tokens = sum(len(t) // 4 for t in observations)

    if current_tokens + new_tokens > MAX_OBSERVATION_TOKENS:
        return {
            "warning": f"Entity approaching token limit ({current_tokens + new_tokens}/{MAX_OBSERVATION_TOKENS})",
            "suggestion": "Consider creating related entities or consolidating observations"
        }
    # proceed normally...
```

**File**: `src/mnemograph/engine.py`

---

## 7. VCS Degradation Documentation (Low)

**Issue**: VCS features degraded with SQLite but not clearly documented.

**Fix**: Update server.py tool descriptions:
```python
# show tool
"NOTE: Historical state (refs other than HEAD) not available with SQLite. Use get_state_at() instead."

# diff tool
"NOTE: Git-based diff unavailable. Use diff_timerange() for event-based comparison."
```

**File**: `src/mnemograph/server.py` (tool descriptions)

---

## 8. Schema Version Check (Low)

**Issue**: No way to detect schema changes requiring migration.

**Fix**: Add version table with minimal migration check:
```python
def _init_db(self):
    conn.execute("CREATE TABLE IF NOT EXISTS schema_version (version INTEGER)")
    version = conn.execute("SELECT version FROM schema_version").fetchone()
    if version is None:
        conn.execute("INSERT INTO schema_version VALUES (1)")
    elif version[0] < CURRENT_VERSION:
        self._migrate(version[0])
```

**File**: `src/mnemograph/events.py`

---

## Execution Order

1. **Similarity fix** (critical, affects duplicate detection)
2. **SQLite concurrency** (high, one-liner)
3. **Compaction safety** (critical, prevents data loss)
4. **Vector invalidation** (high, fixes reload)
5. **clear_graph confirmation** (medium)
6. **Observation limits** (medium)
7. **VCS docs** (low)
8. **Schema version** (low)

## Testing

After each fix:
```bash
uv run python -m pytest -x -q
```

Stress test after similarity fix:
```python
# Create 100 similar entities, verify no false duplicates
for i in range(100):
    engine.create_entities([{"name": f"Concept_{i}", "type": "concept"}])
assert len(engine.state.entities) == 100
```
