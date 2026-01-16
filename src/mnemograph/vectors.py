"""Vector index for semantic search using sqlite-vec and sentence-transformers.

Uses shared SQLite database (mnemograph.db) for both events and vectors.
"""

from __future__ import annotations

import hashlib
import logging
import sqlite3
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .models import Entity

logger = logging.getLogger(__name__)


class VectorStatus(Enum):
    """Status of the vector search subsystem."""

    READY = "ready"
    DEGRADED = "degraded"  # Extension loaded but embedding model failed
    UNAVAILABLE = "unavailable"  # Extension failed to load


@dataclass
class VectorHealth:
    """Health status of the vector index."""

    status: VectorStatus
    error: str | None = None
    embedding_model: str | None = None
    dimension: int | None = None


def _stable_hash(text: str) -> str:
    """Deterministic hash for change detection.

    Uses SHA-256 instead of Python's hash() which is randomized
    by default (PYTHONHASHSEED) for security reasons.
    """
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


class VectorIndex:
    """Vector similarity search backed by sqlite-vec.

    Can use either its own connection or a shared connection from EventStore.
    """

    def __init__(
        self,
        conn: sqlite3.Connection | None = None,
        model_name: str = "all-MiniLM-L6-v2",
    ):
        """Initialize vector index.

        Args:
            conn: Shared SQLite connection (from EventStore). If None, vector
                  search will be unavailable until set_connection() is called.
            model_name: Sentence transformer model to use for embeddings.
        """
        self._model_name = model_name
        self._model = None
        self._dims: int | None = None
        self._conn = conn
        self._owns_connection = False  # We don't own the shared connection

        # Track initialization status
        self.health = VectorHealth(status=VectorStatus.UNAVAILABLE)
        self._extension_loaded = False
        self._model_loaded = False
        self._tables_initialized = False

        # If connection provided, try to load extension (but not model)
        if self._conn is not None:
            self._try_load_extension()

    def set_connection(self, conn: sqlite3.Connection) -> None:
        """Set the shared database connection."""
        self._conn = conn
        self._owns_connection = False
        self._extension_loaded = False
        self._try_load_extension()

    def _try_load_extension(self) -> bool:
        """Try to load sqlite-vec extension. Returns True on success.

        NOTE: This does NOT load the embedding model or create tables.
        Those happen lazily on first search/index operation to avoid
        the 2-3 second cold start delay from sentence-transformers.
        """
        if self._extension_loaded:
            return True

        if self._conn is None:
            self.health = VectorHealth(
                status=VectorStatus.UNAVAILABLE,
                error="No database connection",
            )
            return False

        try:
            import sqlite_vec

            self._conn.enable_load_extension(True)
            sqlite_vec.load(self._conn)
            self._conn.enable_load_extension(False)
            self._extension_loaded = True

            # Check if tables already exist (from previous session)
            self._tables_initialized = self._check_tables_exist()

            # Extension loaded, model not yet loaded (lazy)
            self.health = VectorHealth(
                status=VectorStatus.DEGRADED,
                error="Embedding model loads on first search",
            )
            return True
        except (ImportError, OSError, AttributeError) as e:
            self.health = VectorHealth(
                status=VectorStatus.UNAVAILABLE,
                error=f"sqlite-vec extension failed: {e}",
            )
            logger.warning(f"Vector search unavailable: {e}")
            return False

    def _check_tables_exist(self) -> bool:
        """Check if vector tables already exist in database."""
        import sqlite3

        if self._conn is None:
            return False
        try:
            result = self._conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='entity_meta'"
            ).fetchone()
            return result is not None
        except sqlite3.Error as e:
            logger.debug(f"Table check failed: {e}")
            return False

    def _try_load_model(self) -> bool:
        """Try to load embedding model. Returns True on success."""
        if self._model_loaded:
            return True

        try:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self._model_name)
            self._dims = self._model.get_sentence_embedding_dimension()
            self._model_loaded = True

            # Update health to ready
            self.health = VectorHealth(
                status=VectorStatus.READY,
                embedding_model=self._model_name,
                dimension=self._dims,
            )
            return True
        except (ImportError, OSError, RuntimeError) as e:
            self.health = VectorHealth(
                status=VectorStatus.DEGRADED,
                error=f"Embedding model failed: {e}",
            )
            logger.warning(f"Embedding model unavailable: {e}")
            return False

    @property
    def model(self):
        """Lazy-load the embedding model."""
        if not self._model_loaded:
            self._try_load_model()
        return self._model

    @property
    def dims(self) -> int:
        """Get embedding dimensions (loads model if needed)."""
        from .constants import DEFAULT_EMBEDDING_DIMENSION

        if self._dims is None:
            _ = self.model  # Force load
        return self._dims or DEFAULT_EMBEDDING_DIMENSION

    @property
    def conn(self) -> sqlite3.Connection | None:
        """Get database connection."""
        if self._conn is not None and not self._extension_loaded:
            self._try_load_extension()
        return self._conn

    def _init_tables(self):
        """Initialize database tables."""
        if self._conn is None:
            return

        # Vector table for similarity search
        self._conn.execute(f"""
            CREATE VIRTUAL TABLE IF NOT EXISTS entity_vectors
            USING vec0(
                entity_id TEXT PRIMARY KEY,
                embedding FLOAT[{self.dims}]
            )
        """)

        # Metadata table for filtering and deduplication
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS entity_meta (
                entity_id TEXT PRIMARY KEY,
                name TEXT,
                type TEXT,
                text_hash TEXT
            )
        """)
        self._conn.commit()

    def _entity_to_text(self, entity: Entity) -> str:
        """Convert entity to searchable text."""
        obs_text = ". ".join(o.text for o in entity.observations)
        return f"{entity.name} ({entity.type}): {obs_text}"

    def _ensure_tables(self) -> bool:
        """Ensure tables are initialized. Returns True if ready."""
        if self._tables_initialized:
            return True

        if self._conn is None:
            return False

        # Now we need dims, which triggers model loading
        self._init_tables()
        self._tables_initialized = True
        return True

    def index_entity(self, entity: Entity) -> bool:
        """Add or update entity in vector index. Returns True if indexed."""
        # Ensure connection is available
        conn = self.conn
        if conn is None:
            return False

        # Ensure model is loaded (updates health to READY on success)
        model = self.model
        if model is None:
            return False

        # Ensure tables exist (creates them on first index)
        if not self._ensure_tables():
            return False

        text = self._entity_to_text(entity)
        text_hash = _stable_hash(text)  # Deterministic hash

        # Check if already indexed with same content
        existing = conn.execute(
            "SELECT text_hash FROM entity_meta WHERE entity_id = ?", (entity.id,)
        ).fetchone()

        if existing and existing[0] == text_hash:
            return False  # Already indexed, no change

        # Generate embedding (model already validated above)
        embedding = model.encode(text).tolist()

        import sqlite_vec

        # Upsert vector (delete + insert for sqlite-vec)
        conn.execute("DELETE FROM entity_vectors WHERE entity_id = ?", (entity.id,))
        conn.execute(
            "INSERT INTO entity_vectors (entity_id, embedding) VALUES (?, ?)",
            (entity.id, sqlite_vec.serialize_float32(embedding)),
        )

        # Upsert metadata
        conn.execute(
            """
            INSERT OR REPLACE INTO entity_meta (entity_id, name, type, text_hash)
            VALUES (?, ?, ?, ?)
        """,
            (entity.id, entity.name, entity.type, text_hash),
        )

        conn.commit()
        return True

    def remove_entity(self, entity_id: str) -> None:
        """Remove entity from index."""
        conn = self.conn
        if conn is None:
            return

        conn.execute("DELETE FROM entity_vectors WHERE entity_id = ?", (entity_id,))
        conn.execute("DELETE FROM entity_meta WHERE entity_id = ?", (entity_id,))
        conn.commit()

    def reindex_all(self, entities: list[Entity]) -> int:
        """Reindex all entities. Returns count of indexed."""
        indexed = 0
        for entity in entities:
            if self.index_entity(entity):
                indexed += 1
        return indexed

    def invalidate(self) -> None:
        """Invalidate the vector index, forcing reindex on next use.

        Call this after reloading events from disk to ensure vectors
        are rebuilt from the new state.
        """
        conn = self.conn
        if conn is None:
            return

        # Clear all vector data
        import sqlite3

        try:
            conn.execute("DELETE FROM entity_vectors")
            conn.execute("DELETE FROM entity_meta")
            conn.commit()
            logger.debug("Vector index invalidated")
        except sqlite3.Error as e:
            logger.warning(f"Failed to invalidate vector index: {e}")

    def search(
        self, query: str, limit: int = 10, type_filter: str | None = None
    ) -> list[tuple[str, float]]:
        """Search for similar entities. Returns (entity_id, similarity_score) tuples.

        Returns empty list if vector search is unavailable.

        NOTE: First search triggers model loading (~2-3s cold start).
        """
        # Ensure connection and model are loaded (lazy init)
        conn = self.conn
        if conn is None:
            logger.debug("Vector search unavailable: no database connection")
            return []

        # Trigger model loading (this updates health status)
        model = self.model
        if model is None:
            logger.debug("Vector search unavailable: no embedding model")
            return []

        # Ensure tables exist
        if not self._ensure_tables():
            logger.debug("Vector search unavailable: tables not initialized")
            return []

        # Now check health after lazy loading
        if self.health.status != VectorStatus.READY:
            logger.debug(
                f"Semantic search unavailable ({self.health.status.value}), returning empty"
            )
            return []

        import sqlite_vec

        query_embedding = model.encode(query).tolist()
        serialized = sqlite_vec.serialize_float32(query_embedding)

        if type_filter:
            # For filtered search, we need to search more and filter post-hoc
            # sqlite-vec requires k= in WHERE clause for KNN, JOINs complicate this
            all_results = conn.execute(
                """
                SELECT entity_id, distance
                FROM entity_vectors
                WHERE embedding MATCH ? AND k = ?
                ORDER BY distance
            """,
                (serialized, limit * 5),
            ).fetchall()  # Fetch more to filter

            # Filter by type using metadata
            filtered = []
            for entity_id, distance in all_results:
                meta = conn.execute(
                    "SELECT type FROM entity_meta WHERE entity_id = ?", (entity_id,)
                ).fetchone()
                if meta and meta[0] == type_filter:
                    filtered.append((entity_id, distance))
                    if len(filtered) >= limit:
                        break
            results = filtered
        else:
            results = conn.execute(
                """
                SELECT entity_id, distance
                FROM entity_vectors
                WHERE embedding MATCH ? AND k = ?
                ORDER BY distance
            """,
                (serialized, limit),
            ).fetchall()

        # Convert distance to similarity (1 / (1 + distance))
        return [(r[0], 1.0 / (1.0 + r[1])) for r in results]

    def text_similarity(self, text1: str, text2: str) -> float:
        """Compute cosine similarity between two text strings.

        Returns 0.0 if embeddings unavailable.

        Args:
            text1: First text to compare
            text2: Second text to compare

        Returns:
            Cosine similarity score 0.0-1.0
        """
        model = self.model
        if model is None:
            return 0.0

        try:
            import numpy as np

            # Encode both texts
            emb1 = model.encode(text1)
            emb2 = model.encode(text2)

            # Cosine similarity
            dot = np.dot(emb1, emb2)
            norm1 = np.linalg.norm(emb1)
            norm2 = np.linalg.norm(emb2)

            if norm1 == 0 or norm2 == 0:
                return 0.0

            return float(dot / (norm1 * norm2))
        except (RuntimeError, ValueError, TypeError) as e:
            logger.debug(f"Text similarity computation failed: {e}")
            return 0.0

    def close(self) -> None:
        """Close database connection (only if we own it)."""
        if self._conn and self._owns_connection:
            self._conn.close()
            self._conn = None
