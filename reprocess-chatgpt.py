#!/usr/bin/env python3
"""
reprocess-chatgpt.py — Rewrite dry ChatGPT summaries into rich field notes

Reads raw_excerpt from existing entries and rewrites content to preserve
the texture of discovery — questions asked, surprises found, voice preserved.

Usage:
    # Dry run: preview 10 random rewrites
    python3 reprocess-chatgpt.py --dry-run --sample 10

    # Full run: rewrite all 7,440 ChatGPT entries
    python3 reprocess-chatgpt.py --run

    # Resume after interruption
    python3 reprocess-chatgpt.py --run --resume

Requires: OPENAI_API_KEY environment variable
"""

import json
import sqlite3
import os
import sys
import time
import argparse
import numpy as np
from datetime import datetime, timezone

# ═══════════════════════════════════════
# Rewrite Prompt
# ═══════════════════════════════════════

REWRITE_PROMPT = """You are rewriting a knowledge base entry to make it richer and more alive.

You have:
1. CURRENT ENTRY: A dry summary.
2. RAW CONVERSATION: The original back-and-forth.

Rewrite the entry as a **field note** — dense, specific, interesting to re-read later.

Good examples:
- "Tried generating animated GIFs with pentagons distributed in a circle. The trick: raise random radius to a power n that sweeps 0.1→10→0.1 logarithmically — low n clusters everything at the edge, high n packs the center. 500 pentagons per frame, 20 steps. The density gradient creates a breathing effect."
- "Mike and Walt first meet in Breaking Bad S2E2 'Grilled' (March 2009) — Tuco kidnaps Walt and Jesse, and Mike shows up as Gus's fixer. Their dynamic starts as mutual suspicion, not the grudging respect that comes later."
- "Hubble tension: CMB gives ~67 km/s/Mpc, supernovae give ~73. The gap might come from peculiar velocities — galaxies riding bulk flows that skew the local measurement. Not settled yet."

Bad patterns to AVOID:
- "Starting from a question about X, the conversation revealed..." (NEVER use this)
- "The conversation explored..." / "This discussion covered..."
- "This insight highlighted..." / "Understanding this helps..."
- "Interestingly,..." / "Fascinatingly,..." / "It turns out that..."
- Any sentence that comments on the conversation rather than recording knowledge

Rules:
- Lead with the THING, not with how you learned it. Jump straight into content.
- 2-5 sentences. Every sentence should contain a fact, detail, or insight — zero filler.
- Use specific numbers, names, tools, techniques. "gpt-4o-mini" not "an AI model."
- For creative/fiction: capture the PREMISE and what made it distinctive, not a plot recap.
- For builds: WHAT was built, with WHAT, and what was hard.
- For lookups: state the actual facts. Period.
- If the raw conversation is too thin to improve on, return the current entry unchanged.
- Vary your sentence openings. No two entries should read alike.

Return ONLY the rewritten text. No JSON, no markdown, no preamble."""


def rewrite_entry(current_content, raw_excerpt, api_key, model="gpt-4o-mini"):
    """Send current + raw to OpenAI, get rewritten content back."""
    import httpx

    # Truncate raw excerpt if massive
    raw = raw_excerpt
    if len(raw) > 6000:
        raw = raw[:6000] + "\n[...truncated]"

    user_msg = f"""CURRENT ENTRY:
{current_content}

RAW CONVERSATION:
{raw}"""

    try:
        resp = httpx.post(
            "https://api.openai.com/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": model,
                "temperature": 0.4,
                "max_tokens": 500,
                "messages": [
                    {"role": "system", "content": REWRITE_PROMPT},
                    {"role": "user", "content": user_msg},
                ],
            },
            timeout=30.0,
        )
        resp.raise_for_status()
        data = resp.json()
        text = data["choices"][0]["message"]["content"].strip()
        tokens = data.get("usage", {}).get("total_tokens", 0)
        return text, tokens
    except Exception as e:
        return None, 0


def main():
    parser = argparse.ArgumentParser(description="Rewrite dry ChatGPT entries into rich field notes")
    parser.add_argument("--dry-run", action="store_true", help="Preview rewrites without saving")
    parser.add_argument("--run", action="store_true", help="Actually rewrite and save")
    parser.add_argument("--sample", type=int, default=10, help="Number of entries to sample in dry run")
    parser.add_argument("--resume", action="store_true", help="Skip already-rewritten entries")
    parser.add_argument("--db", default=os.path.expanduser("~/cortex/cortex.db"))
    parser.add_argument("--model", default="gpt-4o-mini")
    parser.add_argument("--batch-size", type=int, default=50, help="Commit every N entries")
    args = parser.parse_args()

    if not args.dry_run and not args.run:
        parser.print_help()
        print("\nSpecify --dry-run or --run")
        sys.exit(1)

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: OPENAI_API_KEY not set")
        sys.exit(1)

    conn = sqlite3.connect(args.db, timeout=30)
    conn.execute("PRAGMA journal_mode=WAL")

    # Check if we have the reprocessed marker column
    cols = [r[1] for r in conn.execute("PRAGMA table_info(entries)").fetchall()]
    if "reprocessed_v3" not in cols:
        conn.execute("ALTER TABLE entries ADD COLUMN reprocessed_v3 INTEGER DEFAULT 0")
        conn.commit()
        print("Added reprocessed_v3 column")

    # Count targets
    if args.resume:
        total = conn.execute(
            "SELECT COUNT(*) FROM entries WHERE source = 'external:chatgpt:v2' AND raw_excerpt IS NOT NULL AND raw_excerpt != '' AND reprocessed_v3 = 0"
        ).fetchone()[0]
    else:
        total = conn.execute(
            "SELECT COUNT(*) FROM entries WHERE source = 'external:chatgpt:v2' AND raw_excerpt IS NOT NULL AND raw_excerpt != ''"
        ).fetchone()[0]

    print(f"Target entries: {total}")

    if args.dry_run:
        # Random sample
        rows = conn.execute(
            f"SELECT id, content, raw_excerpt FROM entries WHERE source = 'external:chatgpt:v2' AND raw_excerpt IS NOT NULL AND raw_excerpt != '' ORDER BY RANDOM() LIMIT {args.sample}"
        ).fetchall()

        total_tokens = 0
        for i, (eid, content, raw) in enumerate(rows):
            print(f"\n{'='*70}")
            print(f"[{i+1}/{len(rows)}] Entry {eid}")
            print(f"{'='*70}")
            print(f"\nBEFORE ({len(content)} chars):")
            print(content[:400])
            print(f"\nRAW EXCERPT ({len(raw)} chars):")
            print(raw[:400])

            rewritten, tokens = rewrite_entry(content, raw, api_key, args.model)
            total_tokens += tokens

            if rewritten:
                print(f"\nAFTER ({len(rewritten)} chars):")
                print(rewritten[:400])
            else:
                print("\n[FAILED]")

        # Cost estimate
        avg_tokens = total_tokens / max(1, len(rows))
        est_total_tokens = avg_tokens * total
        est_cost = est_total_tokens * 0.30 / 1_000_000  # blended gpt-4o-mini price
        print(f"\n{'='*70}")
        print(f"DRY RUN COMPLETE")
        print(f"Entries sampled: {len(rows)}")
        print(f"Tokens used: {total_tokens:,}")
        print(f"Avg tokens/entry: {avg_tokens:.0f}")
        print(f"Est. full run ({total} entries): {est_total_tokens:,.0f} tokens ≈ ${est_cost:.2f}")
        print(f"{'='*70}")

    elif args.run:
        # Load embedding model
        print("Loading embedding model...")
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer("all-MiniLM-L6-v2")
        print("Model loaded")

        if args.resume:
            query = "SELECT id, content, raw_excerpt FROM entries WHERE source = 'external:chatgpt:v2' AND raw_excerpt IS NOT NULL AND raw_excerpt != '' AND reprocessed_v3 = 0 ORDER BY timestamp"
        else:
            query = "SELECT id, content, raw_excerpt FROM entries WHERE source = 'external:chatgpt:v2' AND raw_excerpt IS NOT NULL AND raw_excerpt != '' ORDER BY timestamp"

        rows = conn.execute(query).fetchall()
        print(f"Processing {len(rows)} entries...")

        processed = 0
        failed = 0
        total_tokens = 0
        start = time.time()

        for i, (eid, content, raw) in enumerate(rows):
            rewritten, tokens = rewrite_entry(content, raw, api_key, args.model)
            total_tokens += tokens

            if rewritten and len(rewritten) > 20:
                # Re-embed
                vec = model.encode(rewritten, normalize_embeddings=True)
                blob = vec.astype(np.float32).tobytes()

                # Update content + embedding + mark as reprocessed
                conn.execute(
                    "UPDATE entries SET content = ?, embedding = ?, reprocessed_v3 = 1 WHERE id = ?",
                    (rewritten, blob, eid)
                )

                # Update FTS
                conn.execute(
                    "DELETE FROM entries_fts WHERE rowid = (SELECT rowid FROM entries WHERE id = ?)",
                    (eid,)
                )
                conn.execute(
                    "INSERT INTO entries_fts(rowid, content, tags, source) SELECT rowid, content, tags, source FROM entries WHERE id = ?",
                    (eid,)
                )

                # Delete enrichment so worker re-enriches with new content
                conn.execute("DELETE FROM enrichments WHERE entry_id = ?", (eid,))

                processed += 1
            else:
                # Mark as processed even if failed (don't retry)
                conn.execute("UPDATE entries SET reprocessed_v3 = 1 WHERE id = ?", (eid,))
                failed += 1

            # Commit in batches
            if (i + 1) % args.batch_size == 0:
                conn.commit()
                elapsed = time.time() - start
                rate = (i + 1) / elapsed
                remaining = (len(rows) - i - 1) / rate if rate > 0 else 0
                est_cost = total_tokens * 0.30 / 1_000_000
                print(f"  [{i+1}/{len(rows)}] {processed} rewritten, {failed} failed | "
                      f"{total_tokens:,} tokens (~${est_cost:.2f}) | "
                      f"{rate:.1f}/s | ETA {remaining:.0f}s")

            # Small delay
            time.sleep(0.05)

        conn.commit()
        elapsed = time.time() - start
        est_cost = total_tokens * 0.30 / 1_000_000

        print(f"\n{'='*70}")
        print(f"REPROCESS COMPLETE in {elapsed:.0f}s")
        print(f"  Rewritten:   {processed}")
        print(f"  Failed:      {failed}")
        print(f"  Tokens:      {total_tokens:,}")
        print(f"  Cost:        ~${est_cost:.2f}")
        print(f"  Enrichments deleted (will re-enrich): {processed}")
        print(f"{'='*70}")

    conn.close()


if __name__ == "__main__":
    main()
