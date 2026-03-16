"""
Cortex MCP Server
Persistent conversation memory for Claude with semantic search.

Run: python3 cortex_server.py
Listens on port 8080 (iptables redirects 80 → 8080)
Cloudflared tunnel provides HTTPS endpoint for Claude.ai
Viz available at /viz (or https://cortex.fahrenheitrequited.dev/viz)

Environment variables:
  CORTEX_DB         — Path to SQLite database (default: ~/cortex/cortex.db)
  CORTEX_PORT       — Port to listen on (default: 8080)
  CORTEX_READ_ONLY  — Set to 1/true/yes to disable all write operations
"""

import json
import sqlite3
import os
import time
import numpy as np
import asyncio
import threading
import httpx
from datetime import datetime, timezone
from typing import Optional, List
from contextlib import asynccontextmanager

from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel, Field, ConfigDict
from sentence_transformers import SentenceTransformer

# === Config ===

DB_PATH = os.environ.get("CORTEX_DB", os.path.expanduser("~/cortex/cortex.db"))
PORT = int(os.environ.get("CORTEX_PORT", "8080"))
READ_ONLY = os.environ.get("CORTEX_READ_ONLY", "").lower() in ("1", "true", "yes")
EMBED_MODEL = "all-MiniLM-L6-v2"
EMBED_DIM = 384
VIZ_HTML_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "viz.html")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
OPENAI_CHAT_MODEL = "gpt-4o-mini"
OPENAI_EMBED_MODEL = "text-embedding-3-small"
OPENAI_EMBED_DIM = 1536
ENRICHMENT_BATCH_SIZE = 5  # process this many at a time
ENRICHMENT_INTERVAL = 10   # seconds between checking for unprocessed entries

# === Global model reference (set in lifespan) ===

_model = None

# === Database ===

def init_db():
    """Initialize SQLite database with entries table and embeddings column."""
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS entries (
            id TEXT PRIMARY KEY,
            timestamp TEXT NOT NULL,
            content TEXT NOT NULL,
            tags TEXT NOT NULL DEFAULT '[]',
            source TEXT NOT NULL DEFAULT 'unknown'
        )
    """)
    # Add columns if they don't exist (upgrade path)
    cols = [row[1] for row in conn.execute("PRAGMA table_info(entries)").fetchall()]
    if "embedding" not in cols:
        conn.execute("ALTER TABLE entries ADD COLUMN embedding BLOB")
    if "entry_type" not in cols:
        conn.execute("ALTER TABLE entries ADD COLUMN entry_type TEXT DEFAULT 'note'")
    if "temperature" not in cols:
        conn.execute("ALTER TABLE entries ADD COLUMN temperature REAL DEFAULT 1.0")

    # Edges table for auto-linked entries (similarity, co-tag, etc.)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS edges (
            id TEXT PRIMARY KEY,
            source_id TEXT NOT NULL REFERENCES entries(id) ON DELETE CASCADE,
            target_id TEXT NOT NULL REFERENCES entries(id) ON DELETE CASCADE,
            weight REAL NOT NULL,
            edge_type TEXT DEFAULT 'similar',
            auto_created INTEGER DEFAULT 1,
            created_at TEXT DEFAULT (datetime('now')),
            UNIQUE(source_id, target_id)
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_edges_source ON edges(source_id)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_edges_target ON edges(target_id)")

    conn.execute("""
        CREATE VIRTUAL TABLE IF NOT EXISTS entries_fts
        USING fts5(content, tags, source)
    """)

    # Enrichments table for LLM-extracted entities and relationships
    conn.execute("""
        CREATE TABLE IF NOT EXISTS enrichments (
            entry_id TEXT PRIMARY KEY REFERENCES entries(id) ON DELETE CASCADE,
            entities TEXT NOT NULL DEFAULT '[]',
            relationships TEXT NOT NULL DEFAULT '[]',
            summary TEXT NOT NULL DEFAULT '',
            digest TEXT NOT NULL DEFAULT '',
            openai_embedding BLOB,
            processed_at TEXT NOT NULL
        )
    """)
    # Add digest column if missing (upgrade path)
    enrich_cols = [row[1] for row in conn.execute("PRAGMA table_info(enrichments)").fetchall()]
    if "digest" not in enrich_cols:
        conn.execute("ALTER TABLE enrichments ADD COLUMN digest TEXT NOT NULL DEFAULT ''")

    # Bookmarks table for viz saved views
    conn.execute("""
        CREATE TABLE IF NOT EXISTS bookmarks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            label TEXT NOT NULL,
            entry_id TEXT,
            search_query TEXT,
            breadcrumb TEXT,
            view_members TEXT,
            created_at TEXT DEFAULT (datetime('now'))
        )
    """)
    # Add view_members column if missing (upgrade path)
    try:
        conn.execute("SELECT view_members FROM bookmarks LIMIT 0")
    except Exception:
        conn.execute("ALTER TABLE bookmarks ADD COLUMN view_members TEXT")

    # Rebuild FTS if empty but entries exist
    count = conn.execute("SELECT COUNT(*) FROM entries").fetchone()[0]
    fts_count = conn.execute("SELECT COUNT(*) FROM entries_fts").fetchone()[0]
    if count > 0 and fts_count == 0:
        _rebuild_fts(conn)
    conn.commit()
    conn.close()

def _rebuild_fts(conn):
    """Rebuild FTS index from entries table."""
    conn.execute("DELETE FROM entries_fts")
    conn.execute("""
        INSERT INTO entries_fts(rowid, content, tags, source)
        SELECT rowid, content, tags, source FROM entries
    """)

def get_db():
    """Get a database connection."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def gen_id():
    """Generate a unique ID."""
    import random
    t = int(time.time() * 1000)
    r = random.randint(0, 99999)
    return f"{t:x}-{r:05x}"

def _check_writable(operation: str):
    """Return an error JSON string if server is in read-only mode, else None."""
    if READ_ONLY:
        return json.dumps({"error": f"Server is in read-only mode — {operation} is disabled"})
    return None

# === Embedding helpers ===

def embed_text(text):
    """Encode text to a 384-dim float32 vector, returned as bytes."""
    global _model
    vec = _model.encode(text, normalize_embeddings=True)
    return vec.astype(np.float32).tobytes()

def bytes_to_vec(b):
    """Convert stored bytes back to numpy array."""
    return np.frombuffer(b, dtype=np.float32)

def cosine_sim(a, b):
    """Cosine similarity between two vectors. Pre-normalized so just dot product."""
    return float(np.dot(a, b))

def backfill_embeddings():
    """Embed any entries that don't have embeddings yet."""
    global _model
    conn = get_db()
    rows = conn.execute("SELECT id, content FROM entries WHERE embedding IS NULL").fetchall()
    if not rows:
        conn.close()
        return 0
    count = 0
    for row in rows:
        blob = embed_text(row["content"])
        conn.execute("UPDATE entries SET embedding = ? WHERE id = ?", (blob, row["id"]))
        count += 1
    conn.commit()
    conn.close()
    return count

# === Enrichment Pipeline ===

EXTRACTION_PROMPT = """Analyze this knowledge base entry and extract structured information.

ENTRY:
{content}

TAGS: {tags}

Respond with ONLY valid JSON (no markdown, no backticks):
{{
  "summary": "A concise one-line summary (max 80 chars) suitable as a label in a visualization",
  "digest": "A 2-4 sentence narrative description of what this entry captures, why it matters, and what context it connects to. Write as if helping someone remember this knowledge months later. Be specific and concrete — names, versions, decisions, not abstractions.",
  "entry_type": "one of: decision, insight, bug, feature, architecture, context, skill, monologue, research, personal, reference, note",
  "entities": ["list", "of", "key", "entities", "mentioned"],
  "relationships": [
    {{"entity": "entity_name", "type": "relationship_type"}}
  ]
}}

Rules for entities:
- Extract people, projects, technologies, concepts, places, organizations
- Normalize names (e.g. "JS" → "javascript", "ML" → "machine-learning")
- Lowercase everything
- 3-15 entities typically

Rules for relationships:
- Types should be verbs/prepositions: "discusses", "uses", "implements", "references", "compares", "critiques", "built-with", "related-to", "depends-on"
- Each relationship links this entry TO an entity
- Focus on the most meaningful connections

Rules for summary:
- Max 80 characters
- Should capture the core topic, not just repeat tags
- Should be useful as a node label in a graph visualization

Rules for digest:
- 2-4 sentences, vivid and specific
- Include what decision was made, what problem was solved, or what was learned
- Mention specific names, ports, files, tools — not vague references
- Write as if briefing a future agent who needs to pick up this thread

Rules for entry_type:
- decision: a choice was made between alternatives
- insight: a realization or pattern recognized
- bug: a problem identified or fixed
- feature: something built or planned
- architecture: system design, infrastructure
- context: background information, personal details
- skill: reusable workflow or technique
- monologue: experiential reflection, inner state
- research: investigation, analysis, comparison
- personal: appointments, relationships, life events
- reference: documentation, specs, links
- note: default, general knowledge"""


async def openai_chat(messages, model=None):
    """Call OpenAI chat completions API."""
    model = model or OPENAI_CHAT_MODEL
    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.post(
            "https://api.openai.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"},
            json={"model": model, "messages": messages, "temperature": 0.2, "max_tokens": 1000}
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]


async def openai_embed(texts):
    """Get embeddings from OpenAI API. Returns list of vectors."""
    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.post(
            "https://api.openai.com/v1/embeddings",
            headers={"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"},
            json={"model": OPENAI_EMBED_MODEL, "input": texts}
        )
        resp.raise_for_status()
        data = resp.json()["data"]
        return [np.array(d["embedding"], dtype=np.float32) for d in data]


async def enrich_entry(entry_id, content, tags):
    """Extract entities/relationships and compute OpenAI embedding for one entry."""
    tags_str = ", ".join(tags) if tags else "none"
    prompt = EXTRACTION_PROMPT.format(content=content[:3000], tags=tags_str)

    # Entity extraction via GPT-4o-mini
    try:
        raw = await openai_chat([{"role": "user", "content": prompt}])
        # Strip markdown fences if present
        raw = raw.strip()
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[1] if "\n" in raw else raw[3:]
        if raw.endswith("```"):
            raw = raw[:-3]
        raw = raw.strip()
        extracted = json.loads(raw)
        entities = extracted.get("entities", [])
        relationships = extracted.get("relationships", [])
        summary = extracted.get("summary", "")[:80]
        digest = extracted.get("digest", "")
        classified_type = extracted.get("entry_type", "note")
    except Exception as e:
        print(f"[ENRICH] Extraction failed for {entry_id}: {e}")
        entities = []
        relationships = []
        summary = ""
        digest = ""
        classified_type = "note"

    # OpenAI embedding
    openai_blob = None
    try:
        vecs = await openai_embed([content[:8000]])
        openai_blob = vecs[0].tobytes()
    except Exception as e:
        print(f"[ENRICH] Embedding failed for {entry_id}: {e}")

    # Store results
    now = datetime.now(timezone.utc).isoformat()
    conn = get_db()
    try:
        conn.execute("""
            INSERT OR REPLACE INTO enrichments (entry_id, entities, relationships, summary, digest, openai_embedding, processed_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (entry_id, json.dumps(entities), json.dumps(relationships), summary, digest, openai_blob, now))
        # Auto-classify entry_type if still default
        current_type = conn.execute("SELECT entry_type FROM entries WHERE id = ?", (entry_id,)).fetchone()
        if current_type and current_type[0] in (None, 'note'):
            valid_types = {'decision','insight','bug','feature','architecture','context','skill','monologue','research','personal','reference','note'}
            if classified_type in valid_types:
                conn.execute("UPDATE entries SET entry_type = ? WHERE id = ?", (classified_type, entry_id))
        conn.commit()
        print(f"[ENRICH] {entry_id}: {len(entities)} ent, {len(relationships)} rels, type={classified_type}, digest={len(digest)}ch")
    except Exception as e:
        print(f"[ENRICH] DB write failed for {entry_id}: {e}")
    finally:
        conn.close()


def get_unenriched_entries(limit=None):
    """Find entries that haven't been enriched yet."""
    limit = limit or ENRICHMENT_BATCH_SIZE
    conn = get_db()
    rows = conn.execute("""
        SELECT e.id, e.content, e.tags FROM entries e
        LEFT JOIN enrichments en ON e.id = en.entry_id
        WHERE en.entry_id IS NULL
        ORDER BY e.timestamp DESC
        LIMIT ?
    """, (limit,)).fetchall()
    conn.close()
    return [{"id": r["id"], "content": r["content"], "tags": json.loads(r["tags"])} for r in rows]


ENRICHMENT_PARALLEL = 3  # concurrent enrichment tasks

async def process_enrichment_batch():
    """Process a batch of unenriched entries in parallel."""
    if not OPENAI_API_KEY:
        return 0
    entries = get_unenriched_entries()
    if not entries:
        return 0

    # Process in parallel (up to ENRICHMENT_PARALLEL concurrent)
    sem = asyncio.Semaphore(ENRICHMENT_PARALLEL)
    count = 0

    async def _enrich_one(entry):
        nonlocal count
        async with sem:
            try:
                await enrich_entry(entry["id"], entry["content"], entry["tags"])
                count += 1
            except Exception as e:
                print(f"[ENRICH] Error processing {entry['id']}: {e}")

    await asyncio.gather(*[_enrich_one(e) for e in entries])
    return count


async def enrichment_worker():
    """Background worker: enrichment + auto-linking + temperature decay."""
    if not OPENAI_API_KEY:
        print("[ENRICH] No OPENAI_API_KEY set — enrichment disabled")
        return
    print("[ENRICH] Background enrichment worker started")

    # Initial backfill
    total = 0
    while True:
        n = await process_enrichment_batch()
        total += n
        if n == 0:
            break
        print(f"[ENRICH] Backfill progress: {total} entries enriched so far...")
    if total > 0:
        print(f"[ENRICH] Backfill complete: {total} entries enriched")

    # Build initial auto-links after backfill
    try:
        build_auto_links()
        build_co_tag_links()
    except Exception as e:
        print(f"[LINKS] Initial link build failed: {e}")

    # Ongoing monitoring
    cycle = 0
    while True:
        await asyncio.sleep(ENRICHMENT_INTERVAL)
        cycle += 1
        try:
            n = await process_enrichment_batch()
            if n > 0:
                print(f"[ENRICH] Processed {n} new entries")
                # Rebuild links when new entries enriched
                build_auto_links()
        except Exception as e:
            print(f"[ENRICH] Worker error: {e}")

        # Temperature decay every ~100 cycles (~17 min)
        if cycle % 100 == 0:
            try:
                decay_temperatures()
            except Exception as e:
                print(f"[TEMP] Decay error: {e}")


def start_enrichment_worker():
    """Start the enrichment worker in a background thread with its own event loop."""
    def run():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(enrichment_worker())
    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    return thread


# === Auto-Link Builder ===

EDGE_SIMILARITY_THRESHOLD = 0.55  # minimum cosine similarity for auto-linking
EDGE_TOP_K = 5                     # max edges per entry

def build_auto_links(batch_size=500):
    """Build cosine similarity edges between entries, like OpenPhoto's auto-linker.
    Computes top-K neighbors above threshold for each entry.
    Safe to re-run — uses INSERT OR IGNORE with UNIQUE(source_id, target_id)."""
    conn = get_db()
    rows = conn.execute('SELECT id, embedding FROM entries WHERE embedding IS NOT NULL').fetchall()
    if len(rows) < 2:
        conn.close()
        return 0

    ids = [r['id'] for r in rows]
    embs = np.array([bytes_to_vec(r['embedding']) for r in rows])
    # Already L2-normalized from embed_text(), so cosine sim = dot product

    edge_count = 0
    for i in range(0, len(ids), batch_size):
        chunk = embs[i:i+batch_size]
        sims = chunk @ embs.T  # (batch, N) dot product matrix
        edges = []
        for j, row_sims in enumerate(sims):
            global_i = i + j
            row_sims[global_i] = -1  # exclude self
            top_k = np.argpartition(row_sims, -EDGE_TOP_K)[-EDGE_TOP_K:]
            for k in top_k:
                if row_sims[k] >= EDGE_SIMILARITY_THRESHOLD:
                    src, tgt = ids[global_i], ids[int(k)]
                    if src < tgt:  # canonical ordering to avoid duplicates
                        edges.append((gen_id(), src, tgt, float(row_sims[k]), 'similar', 1))
        if edges:
            conn.executemany(
                "INSERT OR IGNORE INTO edges (id, source_id, target_id, weight, edge_type, auto_created) VALUES (?,?,?,?,?,?)",
                edges
            )
            edge_count += len(edges)
    conn.commit()
    total = conn.execute('SELECT COUNT(*) FROM edges').fetchone()[0]
    conn.close()
    print(f"[LINKS] Added {edge_count} edges ({total} total)")
    return edge_count


def build_co_tag_links():
    """Build edges between entries that share tags."""
    conn = get_db()
    rows = conn.execute('SELECT id, tags FROM entries').fetchall()
    conn.close()

    # Build tag → entry_ids index
    tag_index = {}
    for r in rows:
        for tag in json.loads(r['tags']):
            if tag not in tag_index:
                tag_index[tag] = []
            tag_index[tag].append(r['id'])

    edges = []
    seen = set()
    for tag, entry_ids in tag_index.items():
        if len(entry_ids) > 50:  # skip overly common tags
            continue
        for a in entry_ids:
            for b in entry_ids:
                if a < b:
                    pair = (a, b)
                    if pair not in seen:
                        seen.add(pair)
                        edges.append((gen_id(), a, b, 0.4, 'co_tag', 1))

    if edges:
        conn = get_db()
        conn.executemany(
            "INSERT OR IGNORE INTO edges (id, source_id, target_id, weight, edge_type, auto_created) VALUES (?,?,?,?,?,?)",
            edges
        )
        conn.commit()
        conn.close()
    print(f"[LINKS] Added {len(edges)} co-tag edges")
    return len(edges)


# === Temperature Decay ===

TEMP_DECAY_RATE = 0.98     # multiply temperature by this each day
TEMP_FLOOR = 0.1           # minimum temperature
TEMP_VISIT_BOOST = 0.3     # boost when an entry is accessed

def decay_temperatures():
    """Decay all entry temperatures. Run daily."""
    conn = get_db()
    conn.execute(f"UPDATE entries SET temperature = MAX(?, temperature * ?) WHERE temperature > ?",
                 (TEMP_FLOOR, TEMP_DECAY_RATE, TEMP_FLOOR))
    conn.commit()
    affected = conn.total_changes
    conn.close()
    print(f"[TEMP] Decayed {affected} entries")
    return affected

def boost_temperature(entry_id):
    """Boost an entry's temperature when it's accessed."""
    conn = get_db()
    conn.execute("UPDATE entries SET temperature = MIN(2.0, temperature + ?) WHERE id = ?",
                 (TEMP_VISIT_BOOST, entry_id))
    conn.commit()
    conn.close()


# === Input Models ===

VALID_ENTRY_TYPES = {'decision','insight','bug','feature','architecture','context','skill','monologue','research','personal','reference','note'}

class StoreInput(BaseModel):
    """Input for storing a new entry."""
    model_config = ConfigDict(str_strip_whitespace=True, extra='forbid')
    content: str = Field(..., description="The text content to store", min_length=1)
    tags: List[str] = Field(default_factory=list, description="Tags for categorization, e.g. ['mcp', 'architecture', 'bugfix']")
    source: str = Field(default="unknown", description="Source thread or context description, e.g. 'Thread: MCP server design'")
    entry_type: str = Field(default="note", description="Entry type: decision, insight, bug, feature, architecture, context, skill, monologue, research, personal, reference, note")

class SearchInput(BaseModel):
    """Input for searching entries."""
    model_config = ConfigDict(str_strip_whitespace=True, extra='forbid')
    query: str = Field(..., description="Search query — matches against content, tags, and source", min_length=1)
    limit: int = Field(default=10, description="Max results to return", ge=1, le=100)

class SemanticSearchInput(BaseModel):
    """Input for semantic search."""
    model_config = ConfigDict(str_strip_whitespace=True, extra='forbid')
    query: str = Field(..., description="Natural language query — finds entries by meaning, not just keywords", min_length=1)
    limit: int = Field(default=5, description="Max results to return", ge=1, le=50)
    threshold: float = Field(default=0.3, description="Minimum similarity score (0-1). Lower = more results, higher = stricter matches", ge=0.0, le=1.0)

class ListInput(BaseModel):
    """Input for listing entries."""
    model_config = ConfigDict(extra='forbid')
    limit: int = Field(default=20, description="Max entries to return", ge=1, le=100)
    offset: int = Field(default=0, description="Skip this many entries (for pagination)", ge=0)
    tag: Optional[str] = Field(default=None, description="Filter by tag")

class DeleteInput(BaseModel):
    """Input for deleting an entry."""
    model_config = ConfigDict(str_strip_whitespace=True, extra='forbid')
    entry_id: str = Field(..., description="ID of the entry to delete")

class GetInput(BaseModel):
    """Input for getting a single entry."""
    model_config = ConfigDict(str_strip_whitespace=True, extra='forbid')
    entry_id: str = Field(..., description="ID of the entry to retrieve")

class TagUpdateInput(BaseModel):
    """Input for updating tags on an entry."""
    model_config = ConfigDict(str_strip_whitespace=True, extra='forbid')
    entry_id: str = Field(..., description="ID of the entry to update")
    tags: List[str] = Field(..., description="New tags list (replaces existing)")

class UpdateInput(BaseModel):
    """Input for updating an entry's content."""
    model_config = ConfigDict(str_strip_whitespace=True, extra='forbid')
    entry_id: str = Field(..., description="ID of the entry to update")
    content: str = Field(..., description="New content (replaces existing)", min_length=1)
    tags: Optional[List[str]] = Field(default=None, description="New tags (optional — if omitted, tags are unchanged)")
    source: Optional[str] = Field(default=None, description="New source (optional — if omitted, source is unchanged)")

# === Server ===

@asynccontextmanager
async def app_lifespan(server):
    """Initialize database and load embedding model on startup."""
    global _model
    print("Loading embedding model...")
    _model = SentenceTransformer(EMBED_MODEL)
    print(f"Model loaded: {EMBED_MODEL} ({EMBED_DIM}d)")

    init_db()

    # Backfill any entries missing embeddings
    n = backfill_embeddings()
    if n > 0:
        print(f"Backfilled embeddings for {n} entries")

    yield {}

mcp = FastMCP(
    "cortex_mcp",
    lifespan=app_lifespan,
)

# === Tools ===

@mcp.tool(
    name="cortex_store",
    annotations={
        "title": "Store Entry",
        "readOnlyHint": False,
        "destructiveHint": False,
        "idempotentHint": False,
        "openWorldHint": False,
    }
)
async def cortex_store(params: StoreInput) -> str:
    """Store a new entry in Cortex. Use this to save conversation summaries,
    insights, decisions, code snippets, or any information worth remembering.
    Content should be substantive — extract the signal, not the noise.

    Args:
        params: StoreInput with content, tags, and source fields.

    Returns:
        JSON with the created entry's id, timestamp, and confirmation.
    """
    if err := _check_writable("store"):
        return err

    entry_id = gen_id()
    now = datetime.now(timezone.utc).isoformat()
    tags_json = json.dumps([t.lower().strip() for t in params.tags])
    embedding_blob = embed_text(params.content)

    conn = get_db()
    try:
        conn.execute(
            "INSERT INTO entries (id, timestamp, content, tags, source, embedding, entry_type, temperature) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (entry_id, now, params.content, tags_json, params.source, embedding_blob, params.entry_type if params.entry_type in VALID_ENTRY_TYPES else 'note', 1.5)
        )
        conn.execute(
            "INSERT INTO entries_fts(rowid, content, tags, source) SELECT rowid, content, tags, source FROM entries WHERE id = ?",
            (entry_id,)
        )
        conn.commit()
        preview = params.content[:80].replace("\n", " ")
        et = params.entry_type if params.entry_type in VALID_ENTRY_TYPES else 'note'
        print(f"[STORE] {entry_id}  type={et}  [{', '.join(json.loads(tags_json))}]  {preview}...")
        return json.dumps({
            "status": "stored",
            "id": entry_id,
            "timestamp": now,
            "tags": json.loads(tags_json),
            "entry_type": et,
            "content_preview": params.content[:100] + ("..." if len(params.content) > 100 else ""),
            "bytes": len(params.content.encode('utf-8')),
            "embedded": True
        }, indent=2)
    except Exception as e:
        return json.dumps({"error": f"Failed to store: {str(e)}"})
    finally:
        conn.close()


@mcp.tool(
    name="cortex_search",
    annotations={
        "title": "Full-Text Search",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False,
    }
)
async def cortex_search(params: SearchInput) -> str:
    """Full-text keyword search across all Cortex entries. Searches content, tags, and source fields.
    Returns entries ranked by relevance. Use cortex_semantic_search for meaning-based queries.

    Args:
        params: SearchInput with query string and optional limit.

    Returns:
        JSON array of matching entries with id, timestamp, content preview, tags, source.
    """
    conn = get_db()
    try:
        rows = conn.execute("""
            SELECT e.id, e.timestamp, e.content, e.tags, e.source,
                   rank
            FROM entries_fts fts
            JOIN entries e ON e.rowid = fts.rowid
            WHERE entries_fts MATCH ?
            ORDER BY rank
            LIMIT ?
        """, (params.query, params.limit)).fetchall()

        results = []
        for row in rows:
            results.append({
                "id": row["id"],
                "timestamp": row["timestamp"],
                "content": row["content"],
                "tags": json.loads(row["tags"]),
                "source": row["source"],
            })

        print(f"[SEARCH] query='{params.query}' hits={len(results)}")
        return json.dumps({
            "query": params.query,
            "count": len(results),
            "results": results
        }, indent=2)
    except Exception as e:
        print(f"[SEARCH] FAIL query='{params.query}' err={e}")
        return json.dumps({"error": f"Search failed: {str(e)}", "query": params.query})
    finally:
        conn.close()


@mcp.tool(
    name="cortex_semantic_search",
    annotations={
        "title": "Semantic Search",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False,
    }
)
async def cortex_semantic_search(params: SemanticSearchInput) -> str:
    """Search Cortex entries by meaning using vector embeddings.
    Unlike full-text search, this finds conceptually related entries even if
    they don't share keywords. Use for questions like 'what do I know about
    translation quality' or 'anything related to deployment issues'.

    Args:
        params: SemanticSearchInput with natural language query, limit, and similarity threshold.

    Returns:
        JSON array of matching entries ranked by semantic similarity, with scores.
    """
    query_vec = bytes_to_vec(embed_text(params.query))

    conn = get_db()
    try:
        rows = conn.execute(
            "SELECT id, timestamp, content, tags, source, embedding FROM entries WHERE embedding IS NOT NULL"
        ).fetchall()

        # Temporal decay: recent entries get a boost
        # Formula: final_score = similarity * (1 + recency_boost * decay_factor)
        # decay_factor = exp(-age_days / half_life_days)
        import math
        RECENCY_BOOST = 0.15  # max 15% boost for brand-new entries
        HALF_LIFE_DAYS = 30   # boost halves every 30 days
        now_ts = time.time()

        scored = []
        for row in rows:
            entry_vec = bytes_to_vec(row["embedding"])
            raw_sim = cosine_sim(query_vec, entry_vec)
            if raw_sim >= params.threshold:
                # Compute recency boost
                try:
                    entry_ts = datetime.fromisoformat(row["timestamp"]).timestamp()
                    age_days = max(0, (now_ts - entry_ts) / 86400)
                    decay = math.exp(-age_days * 0.693 / HALF_LIFE_DAYS)  # ln(2) ≈ 0.693
                    score = raw_sim * (1 + RECENCY_BOOST * decay)
                except:
                    score = raw_sim
                scored.append({
                    "id": row["id"],
                    "timestamp": row["timestamp"],
                    "content": row["content"],
                    "tags": json.loads(row["tags"]),
                    "source": row["source"],
                    "similarity": round(raw_sim, 4),
                    "boosted_score": round(score, 4),
                })

        scored.sort(key=lambda x: x["boosted_score"], reverse=True)
        results = scored[:params.limit]

        top_score = results[0]["similarity"] if results else 0
        print(f"[SEMANTIC] query='{params.query}' hits={len(results)}/{len(rows)} top={top_score:.3f}")

        return json.dumps({
            "query": params.query,
            "count": len(results),
            "total_entries_searched": len(rows),
            "threshold": params.threshold,
            "results": results
        }, indent=2)
    except Exception as e:
        return json.dumps({"error": f"Semantic search failed: {str(e)}", "query": params.query})
    finally:
        conn.close()


@mcp.tool(
    name="cortex_list",
    annotations={
        "title": "List Entries",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False,
    }
)
async def cortex_list(params: ListInput) -> str:
    """List Cortex entries in reverse chronological order. Supports pagination and tag filtering.

    Args:
        params: ListInput with limit, offset, and optional tag filter.

    Returns:
        JSON with entries array, total count, and pagination info.
    """
    conn = get_db()
    try:
        if params.tag:
            rows = conn.execute("""
                SELECT id, timestamp, content, tags, source
                FROM entries
                WHERE tags LIKE ?
                ORDER BY timestamp DESC
                LIMIT ? OFFSET ?
            """, (f'%"{params.tag.lower()}"%', params.limit, params.offset)).fetchall()
            total = conn.execute(
                "SELECT COUNT(*) FROM entries WHERE tags LIKE ?",
                (f'%"{params.tag.lower()}"%',)
            ).fetchone()[0]
        else:
            rows = conn.execute("""
                SELECT id, timestamp, content, tags, source
                FROM entries
                ORDER BY timestamp DESC
                LIMIT ? OFFSET ?
            """, (params.limit, params.offset)).fetchall()
            total = conn.execute("SELECT COUNT(*) FROM entries").fetchone()[0]

        entries = []
        for row in rows:
            entries.append({
                "id": row["id"],
                "timestamp": row["timestamp"],
                "content": row["content"],
                "tags": json.loads(row["tags"]),
                "source": row["source"],
            })

        return json.dumps({
            "total": total,
            "count": len(entries),
            "offset": params.offset,
            "has_more": total > params.offset + len(entries),
            "entries": entries
        }, indent=2)
    except Exception as e:
        return json.dumps({"error": f"List failed: {str(e)}"})
    finally:
        conn.close()


@mcp.tool(
    name="cortex_get",
    annotations={
        "title": "Get Entry",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False,
    }
)
async def cortex_get(params: GetInput) -> str:
    """Retrieve a single entry by ID.

    Args:
        params: GetInput with entry_id.

    Returns:
        JSON with the full entry, or error if not found.
    """
    conn = get_db()
    try:
        row = conn.execute(
            "SELECT id, timestamp, content, tags, source, entry_type, temperature FROM entries WHERE id = ?",
            (params.entry_id,)
        ).fetchone()
        if not row:
            return json.dumps({"error": f"Entry {params.entry_id} not found"})
        # Boost temperature on access
        boost_temperature(params.entry_id)
        # Get digest from enrichments if available
        enrichment = conn.execute(
            "SELECT summary, digest FROM enrichments WHERE entry_id = ?",
            (params.entry_id,)
        ).fetchone()
        result = {
            "id": row["id"],
            "timestamp": row["timestamp"],
            "content": row["content"],
            "tags": json.loads(row["tags"]),
            "source": row["source"],
            "entry_type": row["entry_type"] or "note",
            "temperature": row["temperature"] or 1.0,
        }
        if enrichment:
            result["summary"] = enrichment["summary"]
            result["digest"] = enrichment["digest"]
        return json.dumps(result, indent=2)
    finally:
        conn.close()


@mcp.tool(
    name="cortex_delete",
    annotations={
        "title": "Delete Entry",
        "readOnlyHint": False,
        "destructiveHint": True,
        "idempotentHint": True,
        "openWorldHint": False,
    }
)
async def cortex_delete(params: DeleteInput) -> str:
    """Delete an entry from Cortex by ID. This is permanent.

    Args:
        params: DeleteInput with entry_id.

    Returns:
        JSON confirmation or error if not found.
    """
    if err := _check_writable("delete"):
        return err

    conn = get_db()
    try:
        row = conn.execute("SELECT rowid, id FROM entries WHERE id = ?", (params.entry_id,)).fetchone()
        if not row:
            return json.dumps({"error": f"Entry {params.entry_id} not found"})
        conn.execute("DELETE FROM entries_fts WHERE rowid = ?", (row["rowid"],))
        conn.execute("DELETE FROM entries WHERE id = ?", (params.entry_id,))
        conn.commit()
        print(f"[DELETE] {params.entry_id}")
        return json.dumps({"status": "deleted", "id": params.entry_id})
    except Exception as e:
        return json.dumps({"error": f"Delete failed: {str(e)}"})
    finally:
        conn.close()


@mcp.tool(
    name="cortex_update_tags",
    annotations={
        "title": "Update Tags",
        "readOnlyHint": False,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False,
    }
)
async def cortex_update_tags(params: TagUpdateInput) -> str:
    """Update the tags on an existing entry. Replaces all existing tags.

    Args:
        params: TagUpdateInput with entry_id and new tags list.

    Returns:
        JSON confirmation with updated tags.
    """
    if err := _check_writable("update_tags"):
        return err

    conn = get_db()
    try:
        row = conn.execute("SELECT rowid, id FROM entries WHERE id = ?", (params.entry_id,)).fetchone()
        if not row:
            return json.dumps({"error": f"Entry {params.entry_id} not found"})
        tags_json = json.dumps([t.lower().strip() for t in params.tags])
        conn.execute("UPDATE entries SET tags = ? WHERE id = ?", (tags_json, params.entry_id))
        conn.execute("DELETE FROM entries_fts WHERE rowid = ?", (row["rowid"],))
        conn.execute("""
            INSERT INTO entries_fts(rowid, content, tags, source)
            SELECT rowid, content, tags, source FROM entries WHERE id = ?
        """, (params.entry_id,))
        conn.commit()
        return json.dumps({"status": "updated", "id": params.entry_id, "tags": json.loads(tags_json)})
    except Exception as e:
        return json.dumps({"error": f"Update failed: {str(e)}"})
    finally:
        conn.close()


@mcp.tool(
    name="cortex_update",
    annotations={
        "title": "Update Entry",
        "readOnlyHint": False,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False,
    }
)
async def cortex_update(params: UpdateInput) -> str:
    """Update the content (and optionally tags/source) of an existing entry.
    Re-embeds the content, rebuilds the FTS index, and marks the entry
    for re-enrichment. Use this to refine a thought rather than
    creating a new entry.

    Args:
        params: UpdateInput with entry_id, new content, and optional tags/source.

    Returns:
        JSON confirmation with updated fields.
    """
    if err := _check_writable("update"):
        return err

    conn = get_db()
    try:
        row = conn.execute(
            "SELECT rowid, id, tags, source FROM entries WHERE id = ?",
            (params.entry_id,)
        ).fetchone()
        if not row:
            return json.dumps({"error": f"Entry {params.entry_id} not found"})

        rowid = row["rowid"]

        # Determine final values
        new_tags_json = (
            json.dumps([t.lower().strip() for t in params.tags])
            if params.tags is not None
            else row["tags"]
        )
        new_source = params.source if params.source is not None else row["source"]

        # Re-embed with new content
        new_embedding = embed_text(params.content)

        # Update entry
        conn.execute(
            "UPDATE entries SET content = ?, tags = ?, source = ?, embedding = ? WHERE id = ?",
            (params.content, new_tags_json, new_source, new_embedding, params.entry_id)
        )

        # Rebuild FTS for this entry
        conn.execute("DELETE FROM entries_fts WHERE rowid = ?", (rowid,))
        conn.execute(
            "INSERT INTO entries_fts(rowid, content, tags, source) VALUES (?, ?, ?, ?)",
            (rowid, params.content, new_tags_json, new_source)
        )

        # Remove old enrichment so the background worker re-processes it
        conn.execute("DELETE FROM enrichments WHERE entry_id = ?", (params.entry_id,))

        conn.commit()

        preview = params.content[:80].replace("\n", " ")
        print(f"[UPDATE] {params.entry_id}  [{', '.join(json.loads(new_tags_json))}]  {preview}...")

        return json.dumps({
            "status": "updated",
            "id": params.entry_id,
            "content_preview": params.content[:100] + ("..." if len(params.content) > 100 else ""),
            "tags": json.loads(new_tags_json),
            "source": new_source,
            "bytes": len(params.content.encode('utf-8')),
            "re_embedded": True,
            "re_enrichment_queued": True,
        }, indent=2)
    except Exception as e:
        return json.dumps({"error": f"Update failed: {str(e)}"})
    finally:
        conn.close()


@mcp.tool(
    name="cortex_export",
    annotations={
        "title": "Export All Entries",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False,
    }
)
async def cortex_export() -> str:
    """Export all Cortex entries as JSON. Use for backups or migration.

    Returns:
        JSON with all entries and metadata (count, total bytes).
    """
    conn = get_db()
    try:
        rows = conn.execute(
            "SELECT id, timestamp, content, tags, source FROM entries ORDER BY timestamp DESC"
        ).fetchall()
        entries = []
        total_bytes = 0
        for row in rows:
            entry = {
                "id": row["id"],
                "timestamp": row["timestamp"],
                "content": row["content"],
                "tags": json.loads(row["tags"]),
                "source": row["source"],
            }
            total_bytes += len(row["content"].encode('utf-8'))
            entries.append(entry)

        return json.dumps({
            "exported_at": datetime.now(timezone.utc).isoformat(),
            "count": len(entries),
            "total_bytes": total_bytes,
            "entries": entries
        }, indent=2)
    finally:
        conn.close()


@mcp.tool(
    name="cortex_stats",
    annotations={
        "title": "Storage Stats",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False,
    }
)
async def cortex_stats() -> str:
    """Get Cortex storage statistics: entry count, total size, all tags, date range.

    Returns:
        JSON with storage statistics.
    """
    conn = get_db()
    try:
        total = conn.execute("SELECT COUNT(*) FROM entries").fetchone()[0]
        if total == 0:
            return json.dumps({"count": 0, "message": "Cortex is empty"})

        rows = conn.execute("SELECT content, tags, timestamp FROM entries ORDER BY timestamp").fetchall()
        embedded = conn.execute("SELECT COUNT(*) FROM entries WHERE embedding IS NOT NULL").fetchone()[0]
        total_bytes = sum(len(r["content"].encode('utf-8')) for r in rows)
        all_tags = set()
        for r in rows:
            for t in json.loads(r["tags"]):
                all_tags.add(t)

        oldest = rows[0]["timestamp"]
        newest = rows[-1]["timestamp"]
        db_size = os.path.getsize(DB_PATH) if os.path.exists(DB_PATH) else 0

        # Edge and enrichment stats
        try:
            edge_count = conn.execute("SELECT COUNT(*) FROM edges").fetchone()[0]
        except:
            edge_count = 0
        try:
            enriched_count = conn.execute("SELECT COUNT(*) FROM enrichments").fetchone()[0]
        except:
            enriched_count = 0
        try:
            type_rows = conn.execute("SELECT entry_type, COUNT(*) as cnt FROM entries GROUP BY entry_type").fetchall()
            type_dist = {(r[0] or "note"): r[1] for r in type_rows}
        except:
            type_dist = {}

        return json.dumps({
            "entries": total,
            "embedded": embedded,
            "enriched": enriched_count,
            "edges": edge_count,
            "content_bytes": total_bytes,
            "content_kb": round(total_bytes / 1024, 1),
            "db_file_bytes": db_size,
            "db_file_kb": round(db_size / 1024, 1),
            "tags": sorted(all_tags),
            "tag_count": len(all_tags),
            "type_distribution": type_dist,
            "oldest_entry": oldest,
            "newest_entry": newest,
            "model": EMBED_MODEL,
            "embedding_dim": EMBED_DIM,
            "read_only": READ_ONLY,
        }, indent=2)
    finally:
        conn.close()


@mcp.tool(
    name="cortex_build_links",
    annotations={
        "title": "Build Auto-Links",
        "readOnlyHint": False,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False,
    }
)
async def cortex_build_links() -> str:
    """Build automatic similarity edges between entries using cosine similarity
    on embeddings. Also builds co-tag edges. Safe to re-run — won't create duplicates.

    Returns:
        JSON with edge counts.
    """
    if err := _check_writable("build_links"):
        return err
    sim_count = build_auto_links()
    tag_count = build_co_tag_links()
    conn = get_db()
    total = conn.execute('SELECT COUNT(*) FROM edges').fetchone()[0]
    conn.close()
    return json.dumps({
        "status": "complete",
        "similarity_edges_added": sim_count,
        "co_tag_edges_added": tag_count,
        "total_edges": total,
    }, indent=2)


@mcp.tool(
    name="cortex_edges",
    annotations={
        "title": "Get Entry Edges",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False,
    }
)
async def cortex_edges(params: GetInput) -> str:
    """Get all edges (connections) for a specific entry. Returns linked entries
    with similarity scores and edge types.

    Args:
        params: GetInput with entry_id.

    Returns:
        JSON with list of connected entries and their edge weights.
    """
    conn = get_db()
    try:
        rows = conn.execute("""
            SELECT e.id, e.weight, e.edge_type,
                   CASE WHEN e.source_id = ? THEN e.target_id ELSE e.source_id END as neighbor_id
            FROM edges e
            WHERE e.source_id = ? OR e.target_id = ?
            ORDER BY e.weight DESC
        """, (params.entry_id, params.entry_id, params.entry_id)).fetchall()

        neighbors = []
        for row in rows:
            neighbor = conn.execute(
                "SELECT id, timestamp, content, tags, source, entry_type FROM entries WHERE id = ?",
                (row['neighbor_id'],)
            ).fetchone()
            if neighbor:
                neighbors.append({
                    "id": neighbor['id'],
                    "content": neighbor['content'][:200],
                    "tags": json.loads(neighbor['tags']),
                    "entry_type": neighbor['entry_type'],
                    "weight": row['weight'],
                    "edge_type": row['edge_type'],
                })
        # Boost temperature on access
        boost_temperature(params.entry_id)

        return json.dumps({
            "entry_id": params.entry_id,
            "edge_count": len(neighbors),
            "neighbors": neighbors
        }, indent=2)
    finally:
        conn.close()


# === ASGI Middleware ===

# Token auth
MCP_TOKEN = os.environ.get("CORTEX_TOKEN", "emc2ymmv")

class HostRewriteMiddleware:
    """Rewrite Host header + check bearer token for MCP access."""
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] in ("http", "websocket"):
            # --- Token check ---
            if MCP_TOKEN:
                from urllib.parse import parse_qs
                qs = parse_qs(scope.get("query_string", b"").decode())
                token_param = qs.get("token", [None])[0]
                auth_header = None
                for k, v in scope.get("headers", []):
                    if k == b"authorization":
                        auth_header = v.decode()
                        break
                bearer = auth_header.replace("Bearer ", "") if auth_header and auth_header.startswith("Bearer ") else None
                if token_param != MCP_TOKEN and bearer != MCP_TOKEN:
                    if scope["type"] == "http":
                        await send({"type": "http.response.start", "status": 401, "headers": [(b"content-type", b"text/plain")]})
                        await send({"type": "http.response.body", "body": b"Unauthorized"})
                        return
                    else:
                        await send({"type": "websocket.close", "code": 4001})
                        return
            # --- Host rewrite ---
            new_headers = []
            for k, v in scope.get("headers", []):
                if k == b"host":
                    new_headers.append((k, f"localhost:{PORT}".encode()))
                else:
                    new_headers.append((k, v))
            scope["headers"] = new_headers
        await self.app(scope, receive, send)


# === Entry Point ===

if __name__ == "__main__":
    print(f"Starting Cortex MCP server on port {PORT}")
    print(f"Database: {DB_PATH}")
    if READ_ONLY:
        print("⚠  READ-ONLY MODE — write operations disabled")
    print(f"Embedding model: {EMBED_MODEL} ({EMBED_DIM}d)")
    print()

    # Load model and init DB directly
    print("Loading embedding model...")
    _model = SentenceTransformer(EMBED_MODEL)
    print(f"Model loaded: {EMBED_MODEL} ({EMBED_DIM}d)")
    init_db()
    n = backfill_embeddings()
    if n > 0:
        print(f"Backfilled embeddings for {n} entries")

    # Check for viz HTML
    if os.path.exists(VIZ_HTML_PATH):
        print(f"Viz HTML: {VIZ_HTML_PATH}")
    else:
        print(f"WARNING: Viz HTML not found at {VIZ_HTML_PATH} — /viz route will return 404")
    print()

    # Start background enrichment worker
    enrichment_thread = start_enrichment_worker()

    import uvicorn
    from starlette.responses import JSONResponse, HTMLResponse
    from starlette.requests import Request

    # --- REST API handlers ---

    ALLOWED_ORIGINS = {
        "https://cortex.fahrenheitrequited.dev",
        "https://autonomous.fahrenheitrequited.dev",
        "https://openmind.fahrenheitrequited.dev",
        "https://om.fahrenheitrequited.dev",
    }

    def cors_response(data, status_code=200, request=None):
        """JSONResponse with restrictive CORS headers."""
        resp = JSONResponse(data, status_code=status_code)
        origin = None
        if request:
            origin = request.headers.get("origin", "")
        if origin in ALLOWED_ORIGINS:
            resp.headers["Access-Control-Allow-Origin"] = origin
        else:
            resp.headers["Access-Control-Allow-Origin"] = "https://cortex.fahrenheitrequited.dev"
        resp.headers["Access-Control-Allow-Methods"] = "GET, POST, DELETE, OPTIONS"
        resp.headers["Access-Control-Allow-Headers"] = "Authorization, Content-Type"
        resp.headers["Vary"] = "Origin"
        return resp

    def check_api_token(request: Request) -> bool:
        """Verify token on REST API requests. Checks query param or Bearer header."""
        token = request.query_params.get("token", "")
        if token == MCP_TOKEN:
            return True
        auth = request.headers.get("authorization", "")
        if auth == f"Bearer {MCP_TOKEN}":
            return True
        return False

    async def serve_viz(request: Request):
        """Serve the Cortex visualization HTML."""
        if not os.path.exists(VIZ_HTML_PATH):
            return HTMLResponse(
                "<h1>Viz not found</h1><p>Place <code>viz.html</code> next to <code>cortex_server.py</code></p>",
                status_code=404
            )
        with open(VIZ_HTML_PATH, "r") as f:
            html = f.read()
        resp = HTMLResponse(html)
        resp.headers["Cache-Control"] = "no-cache"
        return resp

    async def api_semantic(request: Request):
        if request.method == "OPTIONS":
            return cors_response({})
        q = request.query_params.get("q", "")
        limit = int(request.query_params.get("limit", "10"))
        threshold = float(request.query_params.get("threshold", "0.3"))
        if not q:
            return cors_response({"error": "Missing ?q= parameter"}, 400)

        query_vec = bytes_to_vec(embed_text(q))
        conn = get_db()
        rows = conn.execute(
            "SELECT id, timestamp, content, tags, source, embedding FROM entries WHERE embedding IS NOT NULL"
        ).fetchall()
        conn.close()

        scored = []
        for row in rows:
            entry_vec = bytes_to_vec(row["embedding"])
            score = cosine_sim(query_vec, entry_vec)
            if score >= threshold:
                scored.append({
                    "id": row["id"],
                    "timestamp": row["timestamp"],
                    "content": row["content"],
                    "tags": json.loads(row["tags"]),
                    "source": row["source"],
                    "similarity": round(score, 4),
                })
        scored.sort(key=lambda x: x["similarity"], reverse=True)
        return cors_response({"query": q, "count": len(scored[:limit]), "results": scored[:limit]})

    async def api_search(request: Request):
        q = request.query_params.get("q", "")
        limit = int(request.query_params.get("limit", "10"))
        if not q:
            return cors_response({"error": "Missing ?q= parameter"}, 400)

        conn = get_db()
        try:
            rows = conn.execute("""
                SELECT e.id, e.timestamp, e.content, e.tags, e.source
                FROM entries_fts fts JOIN entries e ON e.rowid = fts.rowid
                WHERE entries_fts MATCH ? ORDER BY rank LIMIT ?
            """, (q, limit)).fetchall()
            results = [{"id": r["id"], "timestamp": r["timestamp"], "content": r["content"],
                        "tags": json.loads(r["tags"]), "source": r["source"]} for r in rows]
            return cors_response({"query": q, "count": len(results), "results": results})
        except Exception as e:
            return cors_response({"error": str(e)}, 500)
        finally:
            conn.close()

    async def api_list(request: Request):
        limit = int(request.query_params.get("limit", "20"))
        offset = int(request.query_params.get("offset", "0"))
        tag = request.query_params.get("tag")

        conn = get_db()
        if tag:
            rows = conn.execute("SELECT id, timestamp, content, tags, source FROM entries WHERE tags LIKE ? ORDER BY timestamp DESC LIMIT ? OFFSET ?",
                                (f'%"{tag.lower()}"%', limit, offset)).fetchall()
            total = conn.execute("SELECT COUNT(*) FROM entries WHERE tags LIKE ?", (f'%"{tag.lower()}"%',)).fetchone()[0]
        else:
            rows = conn.execute("SELECT id, timestamp, content, tags, source FROM entries ORDER BY timestamp DESC LIMIT ? OFFSET ?",
                                (limit, offset)).fetchall()
            total = conn.execute("SELECT COUNT(*) FROM entries").fetchone()[0]
        conn.close()

        entries = [{"id": r["id"], "timestamp": r["timestamp"], "content": r["content"],
                    "tags": json.loads(r["tags"]), "source": r["source"]} for r in rows]
        return cors_response({"total": total, "count": len(entries), "entries": entries})

    async def api_stats(request: Request):
        conn = get_db()
        total = conn.execute("SELECT COUNT(*) FROM entries").fetchone()[0]
        if total == 0:
            conn.close()
            return cors_response({"entries": 0})
        embedded = conn.execute("SELECT COUNT(*) FROM entries WHERE embedding IS NOT NULL").fetchone()[0]
        enriched = conn.execute("SELECT COUNT(*) FROM enrichments").fetchone()[0]
        rows = conn.execute("SELECT content, tags, timestamp FROM entries ORDER BY timestamp").fetchall()
        total_bytes = sum(len(r["content"].encode("utf-8")) for r in rows)
        all_tags = set()
        for r in rows:
            for t in json.loads(r["tags"]):
                all_tags.add(t)
        # Edge count
        try:
            edge_count = conn.execute("SELECT COUNT(*) FROM edges").fetchone()[0]
        except:
            edge_count = 0
        # Type distribution
        try:
            type_rows = conn.execute("SELECT entry_type, COUNT(*) as cnt FROM entries GROUP BY entry_type").fetchall()
            type_dist = {r["entry_type"] or "note": r["cnt"] for r in type_rows}
        except:
            type_dist = {}
        conn.close()
        db_size = os.path.getsize(DB_PATH) if os.path.exists(DB_PATH) else 0
        return cors_response({"entries": total, "embedded": embedded, "enriched": enriched,
                             "edges": edge_count, "content_bytes": total_bytes,
                             "type_distribution": type_dist,
                             "db_bytes": db_size, "tags": sorted(all_tags), "oldest": rows[0]["timestamp"],
                             "newest": rows[-1]["timestamp"], "read_only": READ_ONLY})

    async def api_export_viz(request: Request):
        """Export entries with PCA-projected 3D coordinates and enrichment data for visualization."""
        conn = get_db()
        # Check which optional columns exist in entries table
        col_info = conn.execute("PRAGMA table_info(entries)").fetchall()
        col_names = {c["name"] for c in col_info}
        opt_cols = [c for c in ("convo_id", "metadata", "raw_excerpt", "entry_type", "temperature") if c in col_names]
        select_cols = "id, timestamp, content, tags, source, embedding" + (", " + ", ".join(opt_cols) if opt_cols else "")
        rows = conn.execute(
            f"SELECT {select_cols} FROM entries WHERE embedding IS NOT NULL ORDER BY timestamp"
        ).fetchall()

        # Load enrichments (table may not exist)
        try:
            enrichment_rows = conn.execute(
                "SELECT entry_id, entities, relationships, summary, digest, openai_embedding FROM enrichments"
            ).fetchall()
        except sqlite3.OperationalError:
            enrichment_rows = []
        conn.close()

        enrichments = {}
        for er in enrichment_rows:
            enrichments[er["entry_id"]] = {
                "entities": json.loads(er["entities"]),
                "relationships": json.loads(er["relationships"]),
                "summary": er["summary"],
                "digest": er["digest"] if "digest" in er.keys() else "",
                "openai_embedding": er["openai_embedding"],
            }

        if len(rows) < 1:
            return cors_response({"error": "No embedded entries for projection", "count": 0})

        # Determine which embeddings to use for layout
        openai_count = sum(1 for r in rows if r["id"] in enrichments and enrichments[r["id"]]["openai_embedding"])
        use_openai = openai_count == len(rows)

        ids = []
        entries = []
        vecs = []
        for r in rows:
            ids.append(r["id"])
            entry = {
                "id": r["id"],
                "timestamp": r["timestamp"],
                "content": r["content"],
                "tags": json.loads(r["tags"]),
                "source": r["source"],
                "entry_type": r["entry_type"] if "entry_type" in r.keys() else "note",
                "temperature": r["temperature"] if "temperature" in r.keys() else 1.0,
                "convo_id": r["convo_id"] if "convo_id" in r.keys() else None,
                "raw_excerpt": r["raw_excerpt"] if "raw_excerpt" in r.keys() else None,
            }
            # Parse and attach metadata
            raw_meta = r["metadata"] if "metadata" in r.keys() else None
            if raw_meta:
                try:
                    entry["metadata"] = json.loads(raw_meta)
                except (json.JSONDecodeError, TypeError):
                    pass
            # Attach enrichment data if available
            if r["id"] in enrichments:
                en = enrichments[r["id"]]
                entry["entities"] = en["entities"]
                entry["relationships"] = en["relationships"]
                entry["summary"] = en["summary"]
                entry["digest"] = en.get("digest", "")

            entries.append(entry)

            if use_openai:
                vecs.append(np.frombuffer(enrichments[r["id"]]["openai_embedding"], dtype=np.float32))
            else:
                vecs.append(np.frombuffer(r["embedding"], dtype=np.float32))

        mat = np.stack(vecs)

        # PCA → 3D for display
        mean = mat.mean(axis=0)
        centered = mat - mean
        _, _, Vt = np.linalg.svd(centered, full_matrices=False)
        n_components = min(3, Vt.shape[0])
        projected = centered @ Vt[:n_components].T
        # Pad to 3 columns if fewer than 3 components (e.g. only 2 entries)
        if n_components < 3:
            pad = np.zeros((projected.shape[0], 3 - n_components))
            projected = np.hstack([projected, pad])

        # PCA → 16D for semantic clustering (preserves more variance)
        ndim = min(16, Vt.shape[0])
        cluster_coords = centered @ Vt[:ndim].T
        # Normalize cluster coords
        cc_max = np.abs(cluster_coords).max()
        if cc_max > 0:
            cluster_coords = cluster_coords / cc_max

        # Normalize to [-1, 1] range for three.js
        maxval = np.abs(projected).max()
        if maxval > 0:
            projected = projected / maxval

        for i, entry in enumerate(entries):
            entry["x"] = round(float(projected[i, 0]), 4)
            entry["y"] = round(float(projected[i, 1]), 4)
            entry["z"] = round(float(projected[i, 2]), 4)
            entry["cc"] = [round(float(v), 4) for v in cluster_coords[i]]

        # Collect all unique tags for color mapping
        all_tags = set()
        all_entities = set()
        for e in entries:
            for t in e["tags"]:
                all_tags.add(t)
            for ent in e.get("entities", []):
                all_entities.add(ent)

        # Build entity → entry_ids index for cross-referencing
        entity_index = {}
        for e in entries:
            for ent in e.get("entities", []):
                if ent not in entity_index:
                    entity_index[ent] = []
                entity_index[ent].append(e["id"])

        enriched_count = sum(1 for e in entries if "entities" in e)

        # Load edges for visualization
        conn2 = get_db()
        try:
            edge_rows = conn2.execute(
                "SELECT source_id, target_id, weight, edge_type FROM edges ORDER BY weight DESC"
            ).fetchall()
        except sqlite3.OperationalError:
            edge_rows = []
        conn2.close()

        viz_edges = []
        entry_id_set = set(e["id"] for e in entries)
        for er in edge_rows:
            if er["source_id"] in entry_id_set and er["target_id"] in entry_id_set:
                viz_edges.append({
                    "source": er["source_id"],
                    "target": er["target_id"],
                    "weight": er["weight"],
                    "type": er["edge_type"],
                })

        # Type distribution
        type_counts = {}
        for e in entries:
            t = e.get("entry_type", "note") or "note"
            type_counts[t] = type_counts.get(t, 0) + 1

        return cors_response({
            "count": len(entries),
            "enriched": enriched_count,
            "projection": "pca",
            "embedding_source": "openai" if use_openai else "minilm",
            "dimensions": 3,
            "tags": sorted(all_tags),
            "entities": sorted(all_entities),
            "entity_index": entity_index,
            "type_distribution": type_counts,
            "edge_count": len(viz_edges),
            "edges": viz_edges,
            "entries": entries,
        })

    async def api_enrichment_status(request: Request):
        """Get enrichment pipeline status."""
        conn = get_db()
        total = conn.execute("SELECT COUNT(*) FROM entries").fetchone()[0]
        enriched = conn.execute("SELECT COUNT(*) FROM enrichments").fetchone()[0]
        has_openai_embed = conn.execute("SELECT COUNT(*) FROM enrichments WHERE openai_embedding IS NOT NULL").fetchone()[0]
        pending = total - enriched

        samples = conn.execute("""
            SELECT en.entry_id, en.entities, en.relationships, en.summary, en.processed_at
            FROM enrichments en ORDER BY en.processed_at DESC LIMIT 5
        """).fetchall()
        conn.close()

        sample_list = []
        for s in samples:
            sample_list.append({
                "entry_id": s["entry_id"],
                "entities": json.loads(s["entities"]),
                "relationships": json.loads(s["relationships"]),
                "summary": s["summary"],
                "processed_at": s["processed_at"],
            })

        return cors_response({
            "total_entries": total,
            "enriched": enriched,
            "pending": pending,
            "has_openai_embedding": has_openai_embed,
            "openai_key_configured": bool(OPENAI_API_KEY),
            "chat_model": OPENAI_CHAT_MODEL,
            "embed_model": OPENAI_EMBED_MODEL,
            "recent_enrichments": sample_list,
        })

    # ═══ Bookmarks API ═══

    async def api_bookmarks(request: Request):
        """Handle GET (list) and POST (create) for bookmarks."""
        if request.method == "GET":
            conn = get_db()
            rows = conn.execute(
                "SELECT id, label, entry_id, search_query, breadcrumb, view_members, created_at "
                "FROM bookmarks ORDER BY created_at DESC"
            ).fetchall()
            conn.close()
            return cors_response([{
                "id": r["id"], "label": r["label"], "entry_id": r["entry_id"],
                "search_query": r["search_query"], "breadcrumb": r["breadcrumb"],
                "view_members": r["view_members"], "created_at": r["created_at"]
            } for r in rows])

        elif request.method == "POST":
            data = await request.json()
            conn = get_db()
            cur = conn.execute(
                "INSERT INTO bookmarks (label, entry_id, search_query, breadcrumb, view_members) VALUES (?,?,?,?,?)",
                (data.get("label", "Untitled"), data.get("entry_id"),
                 data.get("search_query"), data.get("breadcrumb"), data.get("view_members"))
            )
            conn.commit()
            bid = cur.lastrowid
            conn.close()
            print(f"[BOOKMARK] Created #{bid}: {data.get('label', 'Untitled')}")
            return cors_response({"id": bid, "status": "created"})

        return cors_response({"error": "method not allowed"}, 405)

    async def api_bookmark_delete(request: Request, bid: int):
        """Delete a bookmark by ID."""
        conn = get_db()
        conn.execute("DELETE FROM bookmarks WHERE id=?", (bid,))
        conn.commit()
        conn.close()
        print(f"[BOOKMARK] Deleted #{bid}")
        return cors_response({"status": "deleted"})

    # --- Build combined app ---

    mcp_app = HostRewriteMiddleware(mcp.streamable_http_app())

    class ApiRouter:
        """Route /api/* and /viz to handlers, everything else to MCP."""
        def __init__(self, mcp_app):
            self.mcp_app = mcp_app
            self.routes = {
                "/api/semantic": api_semantic,
                "/api/search": api_search,
                "/api/list": api_list,
                "/api/stats": api_stats,
                "/api/export-viz": api_export_viz,
                "/api/enrichment-status": api_enrichment_status,
                "/api/bookmarks": api_bookmarks,
                "/viz": serve_viz,
                "/": serve_viz,
            }

        async def __call__(self, scope, receive, send):
            if scope["type"] == "lifespan":
                await self.mcp_app(scope, receive, send)
                return
            if scope["type"] == "http":
                path = scope.get("path", "")
                method = scope.get("method", "GET")

                # Handle CORS preflight for all /api/ routes
                if path.startswith("/api/") and method == "OPTIONS":
                    request = Request(scope, receive)
                    response = cors_response({}, request=request)
                    await response(scope, receive, send)
                    return

                # Token auth for all /api/ routes (not /viz)
                if path.startswith("/api/"):
                    request = Request(scope, receive)
                    if not check_api_token(request):
                        response = cors_response({"error": "unauthorized"}, 401, request=request)
                        await response(scope, receive, send)
                        return

                    # Handle DELETE /api/bookmarks/<id>
                    if path.startswith("/api/bookmarks/") and method == "DELETE":
                        try:
                            bid = int(path.split("/")[-1])
                            response = await api_bookmark_delete(request, bid)
                            await response(scope, receive, send)
                            return
                        except (ValueError, IndexError):
                            pass

                    # Exact route matching for /api/*
                    if path in self.routes:
                        response = await self.routes[path](request)
                        await response(scope, receive, send)
                        return

                # Unauthenticated routes: /viz, /
                if path in ("/viz", "/"):
                    request = Request(scope, receive)
                    response = await self.routes[path](request)
                    await response(scope, receive, send)
                    return

            await self.mcp_app(scope, receive, send)

    app = ApiRouter(mcp_app)
    uvicorn.run(app, host="127.0.0.1", port=PORT)
