# Cortex Spec — v2

**Last updated:** 2026-03-15
**Status:** Living document — update with each feature session

---

## What Is Cortex

A persistent knowledge store and semantic memory system for AI agents. Captures conversation summaries, insights, decisions, code snippets, and any information worth remembering across sessions. Provides full-text search, vector-similarity search with temporal decay, LLM-powered enrichment (entity extraction, narrative digests, auto-classification), auto-linking by cosine similarity and co-tag overlap, temperature-based entry salience, and a 3D visualization frontend.

**Production URLs:**
- Main: https://cortex.fahrenheitrequited.dev (port 8080)
- Autonomous: https://autonomous.fahrenheitrequited.dev (port 8082)
- Viz: https://cortex.fahrenheitrequited.dev/viz

## Data Model

### Entries Table
id (TEXT PK) · timestamp (ISO 8601) · content (TEXT) · tags (JSON array) · source (TEXT) · embedding (BLOB 384d) · entry_type (TEXT, v2) · temperature (REAL, v2)

### Entry Types (12)
decision · insight · bug · feature · architecture · context · skill · monologue · research · personal · reference · note (default)

### Enrichments Table
entry_id (FK) · entities (JSON) · relationships (JSON) · summary (≤80 chars) · digest (v2, 2-4 sentence narrative) · openai_embedding (BLOB 1536d) · processed_at

### Edges Table (v2 — new)
id (TEXT PK) · source_id (FK) · target_id (FK) · weight (REAL) · edge_type (similar|co_tag) · auto_created (INTEGER) · created_at. UNIQUE(source_id, target_id).

### FTS5 Index on content, tags, source.

## Dual Instances
Main (8080, ~/cortex/cortex.db): ~8K entries, ~114 MB. Autonomous (8082, ~/cortex/autonomous.db): ~8 MB. Same code, different env vars.

## Embeddings
Local: all-MiniLM-L6-v2 (384d, CPU). OpenAI: text-embedding-3-small (1536d, in enrichments).

## Enrichment Pipeline (v2 enhanced)
3x parallel (asyncio semaphore). Extracts: entities, relationships, summary, digest (v2), entry_type (v2). After backfill: auto-builds links. Every ~17 min: decays temperatures.

## Auto-Link System (v2 — new)
Cosine similarity: top-5 neighbors above 0.55 threshold, batch matrix multiply. Co-tag: weight 0.4, skips tags >50 entries. Both use INSERT OR IGNORE, safe to re-run.

## Temperature System (v2 — new)
Initial: 1.5. Decay: 0.98/day. Floor: 0.1. Visit boost: +0.3. Cap: 2.0. Decays periodically in background worker.

## Temporal Decay on Search (v2 — new)
final_score = similarity × (1 + 0.15 × e^(-age_days × ln2 / 30)). 15% max boost, 30-day half-life. Returns both similarity and boosted_score.

## MCP Tools (12 total, require ?token=emc2ymmv)
Write: cortex_store (accepts entry_type), cortex_update, cortex_update_tags, cortex_delete, cortex_build_links (v2)
Read: cortex_search, cortex_semantic_search (temporal decay), cortex_list, cortex_get (returns type/temp/digest, boosts temp), cortex_edges (v2), cortex_stats (includes edges/types), cortex_export

## REST API (unauthenticated)
/api/stats (v2: edges, types) · /api/search · /api/semantic · /api/list · /api/export-viz (v2: edges, types, digest, temperature) · /api/enrichment-status · /api/bookmarks · /viz

## Security
127.0.0.1 only. Token: emc2ymmv (query param or Bearer header). HostRewriteMiddleware wraps MCP only. CORS: * on API.

## Infrastructure
Server: c-jfischer3. Python: ~/miniconda3/bin/python3. Source: ~/claude/cortex/. Git: requitefahrenheit/cortex, branch v2/enrichment-upgrade. Tunnel: Cloudflare cortex (5382c123).

## Migration v1→v2
Schema: automatic on startup (ALTER TABLE adds entry_type, temperature, digest; CREATE TABLE edges). Re-enrichment: DELETE FROM enrichments then restart (~$1, few hours). Auto-links: built automatically after backfill.

## Version History
v0 (Feb 18): Initial. v1 (Mar 3): Security hardening, first spec. v2 (Mar 15): Entry types, digests, auto-links, temperature, temporal decay, parallel enrichment, cortex_build_links + cortex_edges tools.
