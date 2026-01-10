"""Vector index for semantic search using sqlite-vec and sentence-transformers."""

import sqlite3
from pathlib import Path

import sqlite_vec
from sentence_transformers import SentenceTransformer

from .models import Entity


class VectorIndex:
    """Vector similarity search backed by sqlite-vec."""

    def __init__(self, db_path: Path, model_name: str = "all-MiniLM-L6-v2"):
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._model: SentenceTransformer | None = None
        self._model_name = model_name
        self._dims: int | None = None
        self._conn: sqlite3.Connection | None = None

    @property
    def model(self) -> SentenceTransformer:
        """Lazy-load the embedding model."""
        if self._model is None:
            self._model = SentenceTransformer(self._model_name)
            self._dims = self._model.get_sentence_embedding_dimension()
        return self._model

    @property
    def dims(self) -> int:
        """Get embedding dimensions (loads model if needed)."""
        if self._dims is None:
            _ = self.model  # Force load
        return self._dims  # type: ignore

    @property
    def conn(self) -> sqlite3.Connection:
        """Lazy-load database connection."""
        if self._conn is None:
            self._conn = sqlite3.connect(str(self.db_path))
            self._conn.enable_load_extension(True)
            sqlite_vec.load(self._conn)
            self._conn.enable_load_extension(False)
            self._init_tables()
        return self._conn

    def _init_tables(self):
        """Initialize database tables."""
        # Vector table for similarity search
        self.conn.execute(f"""
            CREATE VIRTUAL TABLE IF NOT EXISTS entity_vectors
            USING vec0(
                entity_id TEXT PRIMARY KEY,
                embedding FLOAT[{self.dims}]
            )
        """)

        # Metadata table for filtering and deduplication
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS entity_meta (
                entity_id TEXT PRIMARY KEY,
                name TEXT,
                type TEXT,
                text_hash TEXT
            )
        """)
        self.conn.commit()

    def _entity_to_text(self, entity: Entity) -> str:
        """Convert entity to searchable text."""
        obs_text = ". ".join(o.text for o in entity.observations)
        return f"{entity.name} ({entity.type}): {obs_text}"

    def index_entity(self, entity: Entity) -> bool:
        """Add or update entity in vector index. Returns True if indexed."""
        text = self._entity_to_text(entity)
        text_hash = str(hash(text))

        # Check if already indexed with same content
        existing = self.conn.execute(
            "SELECT text_hash FROM entity_meta WHERE entity_id = ?",
            (entity.id,)
        ).fetchone()

        if existing and existing[0] == text_hash:
            return False  # Already indexed, no change

        # Generate embedding
        embedding = self.model.encode(text).tolist()

        # Upsert vector (delete + insert for sqlite-vec)
        self.conn.execute("DELETE FROM entity_vectors WHERE entity_id = ?", (entity.id,))
        self.conn.execute(
            "INSERT INTO entity_vectors (entity_id, embedding) VALUES (?, ?)",
            (entity.id, sqlite_vec.serialize_float32(embedding))
        )

        # Upsert metadata
        self.conn.execute("""
            INSERT OR REPLACE INTO entity_meta (entity_id, name, type, text_hash)
            VALUES (?, ?, ?, ?)
        """, (entity.id, entity.name, entity.type, text_hash))

        self.conn.commit()
        return True

    def remove_entity(self, entity_id: str) -> None:
        """Remove entity from index."""
        self.conn.execute("DELETE FROM entity_vectors WHERE entity_id = ?", (entity_id,))
        self.conn.execute("DELETE FROM entity_meta WHERE entity_id = ?", (entity_id,))
        self.conn.commit()

    def reindex_all(self, entities: list[Entity]) -> int:
        """Reindex all entities. Returns count of indexed."""
        indexed = 0
        for entity in entities:
            if self.index_entity(entity):
                indexed += 1
        return indexed

    def search(
        self,
        query: str,
        limit: int = 10,
        type_filter: str | None = None
    ) -> list[tuple[str, float]]:
        """Search for similar entities. Returns (entity_id, similarity_score) tuples."""
        query_embedding = self.model.encode(query).tolist()
        serialized = sqlite_vec.serialize_float32(query_embedding)

        if type_filter:
            # For filtered search, we need to search more and filter post-hoc
            # sqlite-vec requires k= in WHERE clause for KNN, JOINs complicate this
            all_results = self.conn.execute("""
                SELECT entity_id, distance
                FROM entity_vectors
                WHERE embedding MATCH ? AND k = ?
                ORDER BY distance
            """, (serialized, limit * 5)).fetchall()  # Fetch more to filter

            # Filter by type using metadata
            filtered = []
            for entity_id, distance in all_results:
                meta = self.conn.execute(
                    "SELECT type FROM entity_meta WHERE entity_id = ?", (entity_id,)
                ).fetchone()
                if meta and meta[0] == type_filter:
                    filtered.append((entity_id, distance))
                    if len(filtered) >= limit:
                        break
            results = filtered
        else:
            results = self.conn.execute("""
                SELECT entity_id, distance
                FROM entity_vectors
                WHERE embedding MATCH ? AND k = ?
                ORDER BY distance
            """, (serialized, limit)).fetchall()

        # Convert distance to similarity (1 / (1 + distance))
        return [(r[0], 1.0 / (1.0 + r[1])) for r in results]

    def close(self) -> None:
        """Close database connection."""
        if self._conn:
            self._conn.close()
            self._conn = None
