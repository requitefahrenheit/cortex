# Cortex Spec — v2

**Last updated:** 2026-03-15
**Status:** Living document — update with each feature session
**Deployment:** v2 live as of 2026-03-15 20:35 UTC (branch v2/enrichment-upgrade, commits f0e7887 + d0f0530)

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
id (TEXT PK) · source_id (FK) · target_id (FK) · weight (REAL) · edge_type (similar|co_tag) · auto_created (INTEGER) · created_at. UNIQUE(source_id, target_id). Indexes on source_id, target_id.

### FTS5 Index on content, tags, source.

## Current Scale
~9,281 entries · ~134K edges (16K similarity + 126K co-tag) · ~164 MB DB · ~12,847 unique tags · date range 2022-12 to present.

## Dual Instances
Main (8080, ~/cortex/cortex.db): primary store. Autonomous (8082, ~/cortex/autonomous.db): agent-only. Same code (~/claude/cortex/dual-server.py), different env vars (CORTEX_DB, CORTEX_PORT).

## Embeddings
Local: all-MiniLM-L6-v2 (384d, CPU, ~18s cold start). OpenAI: text-embedding-3-small (1536d, stored in enrichments table for future hybrid search).

## Enrichment Pipeline (v2 enhanced)
3x parallel (asyncio.Semaphore). GPT-4o-mini extracts per entry:
- **entities** — named things (people, tools, concepts, ports, files)
- **relationships** — (entity, relation_type) pairs
- **summary** — ≤80 char one-liner
- **digest** (v2) — 2-4 sentence narrative for a future agent: specific names, ports, files, decisions. Not vague.
- **entry_type** (v2) — auto-classifies from 12-type taxonomy, writes back to entries.entry_type if still 'note'

After backfill completes: auto-builds cosine + co-tag links. Every ~17 min: decays temperatures.

Digest quality verified: entries correctly classified (Visit One chapters → monologue/skill, translation workflow → insight, deployment notes → architecture/feature). Digests reference specific names, tools, and context.

## Auto-Link System (v2 — new)
Cosine similarity: top-5 neighbors above 0.55 threshold, batch matrix multiply via numpy. Co-tag: weight 0.4, skips tags with >50 entries to avoid spam edges. Both INSERT OR IGNORE, safe to re-run. Built automatically on server startup and after enrichment backfill.

## Temperature System (v2 — new)
Initial: 1.5. Decay: 0.98/day. Floor: 0.1. Visit boost: +0.3 (on cortex_get, cortex_edges). Cap: 2.0. decay_temperatures() runs in background worker every ~17 min.

## Temporal Decay on Semantic Search (v2 — new)
final_score = similarity × (1 + 0.15 × e^(-age_days × ln2 / 30)). 15% max boost for fresh entries, 30-day half-life. Returns both raw similarity and boosted_score.

## MCP Tools (12 total, require ?token=emc2ymmv)
Write: cortex_store (accepts entry_type), cortex_update, cortex_update_tags, cortex_delete, cortex_build_links (v2)
Read: cortex_search, cortex_semantic_search (temporal decay), cortex_list, cortex_get (returns type/temp/digest, boosts temp), cortex_edges (v2, boosts temp), cortex_stats (includes edges/types), cortex_export

## REST API (unauthenticated, localhost only)
/api/stats (v2: edges, types) · /api/search · /api/semantic · /api/list · /api/export-viz (v2: edges, types, digest, temperature) · /api/enrichment-status · /api/bookmarks · /viz

## Security
All servers bind 127.0.0.1 only (verified 2026-03-15 — zero services on 0.0.0.0). Token: emc2ymmv (query param or Authorization: Bearer header). HostRewriteMiddleware wraps MCP StreamableHTTP endpoint. CORS: * on REST API. Cloudflare tunnel provides the only external access path.

## Infrastructure
Server: c-jfischer3. Python: ~/miniconda3/bin/python3. Source: ~/claude/cortex/dual-server.py (~1,820 lines). Git: requitefahrenheit/cortex, branch v2/enrichment-upgrade. Launch: `at now` with nohup (screen/setsid unreliable through dev tools). Tunnel: Cloudflare named tunnel "cortex" (ID: 5382c123).

## Known Issues
- autonomous instance entry_type field not exposed via MCP (StoreInput on autonomous lacks entry_type param — needs sync)
- VALID_ENTRY_TYPES validation on store is permissive (falls back to 'note' silently)

## Migration v1→v2
Schema: automatic on startup (ALTER TABLE adds entry_type, temperature, digest; CREATE TABLE edges with indexes). Re-enrichment: DELETE FROM enrichments then restart (~$1 GPT-4o-mini, 4-6 hours for ~9K entries). Auto-links: built on startup automatically.

## Version History
v0 (Feb 18): Initial deploy. v1 (Mar 3): Security hardening (localhost bind, token auth), first spec. v2 (Mar 15): Entry types (12), narrative digests, auto-links (134K edges), temperature system, temporal decay on search, parallel enrichment (3x), cortex_build_links + cortex_edges tools. Bugfix: cortex_store NameError (bare entry_type → params.entry_type).
