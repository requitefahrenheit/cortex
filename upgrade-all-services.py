#!/usr/bin/env python3
"""
upgrade-all-services.py — Point all services at Cortex v2

Fixes:
1. DUAL_SERVER path: mcp-server/dual-server.py → cortex/dual-server.py (3 files)
2. Add entry_type to cortex_store tool definitions (2 files)
3. Add cortex_edges + cortex_build_links tool definitions (2 files)
4. Rewrite dispatcher store_in_cortex to use MCP API (1 file)

Dry run by default. Pass --apply to make changes.
"""

import os
import sys
import re
from pathlib import Path

HOME = Path.home()
DRY_RUN = "--apply" not in sys.argv

def patch_file(path, old, new, label):
    """Find old string in file, replace with new. Returns True if patched."""
    p = Path(path)
    if not p.exists():
        print(f"  SKIP {label}: file not found ({path})")
        return False
    content = p.read_text()
    if old not in content:
        if new in content:
            print(f"  OK   {label}: already patched")
            return False
        print(f"  WARN {label}: old string not found")
        return False
    if DRY_RUN:
        print(f"  WOULD patch {label}: {path}")
        return False
    content = content.replace(old, new, 1)
    p.write_text(content)
    print(f"  DONE {label}: {path}")
    return True


def insert_after(path, anchor, insertion, label):
    """Insert text after anchor string. Returns True if inserted."""
    p = Path(path)
    if not p.exists():
        print(f"  SKIP {label}: file not found")
        return False
    content = p.read_text()
    if insertion.strip()[:60] in content:
        print(f"  OK   {label}: already present")
        return False
    if anchor not in content:
        print(f"  WARN {label}: anchor not found")
        return False
    if DRY_RUN:
        print(f"  WOULD insert {label}: {path}")
        return False
    content = content.replace(anchor, anchor + insertion, 1)
    p.write_text(content)
    print(f"  DONE {label}: {path}")
    return True


def main():
    if DRY_RUN:
        print("=== DRY RUN (pass --apply to make changes) ===\n")
    else:
        print("=== APPLYING CHANGES ===\n")

    changes = 0

    # ──────────────────────────────────────────────
    # 1. Fix DUAL_SERVER path (3 files)
    # ──────────────────────────────────────────────
    print("[1] Fix DUAL_SERVER path → ~/claude/cortex/dual-server.py")

    old_path = '"claude" / "mcp-server" / "dual-server.py"'
    new_path = '"claude" / "cortex" / "dual-server.py"'

    for f in [
        HOME / "claude" / "opensquid" / "agent_cortex.py",
        HOME / "claude" / "dispatcher" / "agent_cortex.py",
    ]:
        if patch_file(f, old_path, new_path, f"DUAL_SERVER in {f.name}"):
            changes += 1

    # Daemon has it as _DUAL_SERVER with same path
    daemon = HOME / "claude" / "opensquid" / "daemon" / "daemon-server.py"
    if patch_file(daemon, old_path, new_path, "_DUAL_SERVER in daemon-server.py"):
        changes += 1

    # ──────────────────────────────────────────────
    # 2. Add entry_type to cortex_store tool (2 files)
    # ──────────────────────────────────────────────
    print("\n[2] Add entry_type param to cortex_store tool definition")

    entry_type_prop = '''"entry_type": {"type": "string", "description": "Entry type: decision, insight, bug, feature, architecture, context, skill, monologue, research, personal, reference, note", "default": "note"}'''

    for f in [
        HOME / "claude" / "opensquid" / "daemon" / "daemon-server.py",
        HOME / "claude" / "parity" / "daemon" / "daemon-server.py",
    ]:
        # Find cortex_store tool and add entry_type after source property
        anchor = '"source": {"type": "string", "description": "Source context"}'
        insertion = f",\n                {entry_type_prop}"
        # Only insert if cortex_store is nearby (crude but works)
        if f.exists() and anchor in f.read_text() and 'entry_type' not in f.read_text().split('cortex_store')[1].split('cortex_search')[0] if 'cortex_store' in f.read_text() else True:
            if insert_after(f, anchor, insertion, f"entry_type in {f.parent.parent.name}/{f.name}"):
                changes += 1
        else:
            p = Path(f)
            if p.exists():
                content = p.read_text()
                if 'entry_type' in content and 'cortex_store' in content:
                    print(f"  OK   entry_type in {f.parent.parent.name}/{f.name}: already present")
                else:
                    print(f"  WARN entry_type in {f.parent.parent.name}/{f.name}: needs manual check")

    # ──────────────────────────────────────────────
    # 3. Add cortex_edges + cortex_build_links tools (2 files)
    # ──────────────────────────────────────────────
    print("\n[3] Add cortex_edges and cortex_build_links tool definitions")

    new_tools = '''
    {
        "name": "cortex_edges",
        "description": "Get all edges (connections) for a specific entry. Returns similar entries and co-tag neighbors with weights.",
        "input_schema": {
            "type": "object",
            "properties": {
                "entry_id": {"type": "string", "description": "Entry ID to get edges for"}
            },
            "required": ["entry_id"]
        }
    },
    {
        "name": "cortex_build_links",
        "description": "Build automatic similarity and co-tag edges between entries. Run after bulk imports or to refresh the link graph.",
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": []
        }
    },'''

    for f in [
        HOME / "claude" / "opensquid" / "daemon" / "daemon-server.py",
        HOME / "claude" / "parity" / "daemon" / "daemon-server.py",
    ]:
        if not f.exists():
            print(f"  SKIP {f.parent.parent.name}: file not found")
            continue
        content = f.read_text()
        if 'cortex_edges' in content:
            print(f"  OK   cortex_edges in {f.parent.parent.name}/{f.name}: already present")
            continue
        # Insert after cortex_semantic_search tool definition
        # Find the closing of the semantic_search tool
        anchor = '"name": "cortex_semantic_search"'
        if anchor not in content:
            print(f"  WARN {f.parent.parent.name}: cortex_semantic_search not found")
            continue
        # Find the end of that tool definition block (next '},\n')
        idx = content.index(anchor)
        # Find closing brace of this tool dict
        brace_count = 0
        # Go backwards to find the opening { of this tool
        search_start = content.rfind('{', 0, idx)
        # Now go forward from search_start to find matching }
        pos = search_start
        for i, ch in enumerate(content[search_start:], start=search_start):
            if ch == '{':
                brace_count += 1
            elif ch == '}':
                brace_count -= 1
                if brace_count == 0:
                    pos = i
                    break
        # Insert after closing } of semantic_search tool
        insert_point = pos + 1
        if DRY_RUN:
            print(f"  WOULD insert cortex_edges + cortex_build_links in {f.parent.parent.name}/{f.name}")
        else:
            new_content = content[:insert_point] + new_tools + content[insert_point:]
            f.write_text(new_content)
            changes += 1
            print(f"  DONE cortex_edges + cortex_build_links in {f.parent.parent.name}/{f.name}")

    # ──────────────────────────────────────────────
    # 4. Rewrite dispatcher store_in_cortex to use HTTP API
    # ──────────────────────────────────────────────
    print("\n[4] Rewrite dispatcher store_in_cortex → HTTP API")

    dispatcher_py = HOME / "claude" / "dispatcher" / "dispatcher.py"

    old_store = '''def store_in_cortex(session_result: dict):
    """Write session summary directly to Cortex SQLite DB."""
    try:
        import sqlite3
        db_path = str(CORTEX_DB)
        if not os.path.exists(db_path):
            return

        sid = session_result["session_id"]
        goal = session_result.get("goal", "")
        cost = session_result.get("cost", 0)
        status_str = session_result.get("status", "unknown")
        turns = session_result.get("turns", 0)
        skill_names = session_result.get("skill_names", [])
        snippet = session_result.get("result_snippet", "")

        tags = json.dumps(["dispatcher", sid] + list(skill_names))
        content = (
            f"**Dispatcher session:** {sid}\\n"
            f"**Goal:** {goal}\\n\\n"
            f"**Output:**\\n{snippet[:3000]}\\n\\n"
            f"**Status:** {status_str}  **Cost:** ${cost:.4f}  **Turns:** {turns}"
        )
        entry_id = uuid.uuid4().hex
        ts = datetime.utcnow().isoformat() + "Z"
        source = f"dispatcher/{sid}"

        conn = sqlite3.connect(db_path)
        conn.execute(
            "INSERT INTO entries (id, timestamp, content, tags, source) VALUES (?,?,?,?,?)",
            (entry_id, ts, content, tags, source),
        )
        conn.execute(
            "INSERT INTO entries_fts (content, tags, source) VALUES (?,?,?)",
            (content, tags, source),
        )
        conn.commit()
        conn.close()
        return entry_id
    except Exception as e:
        print(f"[DISPATCHER] Cortex store failed: {e}")
        return None'''

    new_store = '''def store_in_cortex(session_result: dict):
    """Store session summary in Cortex via MCP API (uses running server, gets embeddings + enrichment)."""
    try:
        import urllib.request

        sid = session_result["session_id"]
        goal = session_result.get("goal", "")
        cost = session_result.get("cost", 0)
        status_str = session_result.get("status", "unknown")
        turns = session_result.get("turns", 0)
        skill_names = session_result.get("skill_names", [])
        snippet = session_result.get("result_snippet", "")

        tags = ["dispatcher", sid] + list(skill_names)
        content = (
            f"**Dispatcher session:** {sid}\\n"
            f"**Goal:** {goal}\\n\\n"
            f"**Output:**\\n{snippet[:3000]}\\n\\n"
            f"**Status:** {status_str}  **Cost:** ${cost:.4f}  **Turns:** {turns}"
        )

        # Call Cortex MCP server via JSON-RPC
        payload = json.dumps({
            "jsonrpc": "2.0",
            "id": str(uuid.uuid4()),
            "method": "tools/call",
            "params": {
                "name": "cortex_store",
                "arguments": {
                    "content": content,
                    "tags": tags,
                    "source": f"dispatcher/{sid}",
                    "entry_type": "context"
                }
            }
        }).encode()

        req = urllib.request.Request(
            "http://127.0.0.1:8080/mcp?token=emc2ymmv",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST"
        )
        resp = urllib.request.urlopen(req, timeout=15)
        result = json.loads(resp.read())
        print(f"[DISPATCHER] Stored in Cortex via MCP: {sid}")
        return result
    except Exception as e:
        print(f"[DISPATCHER] Cortex store failed: {e}")
        return None'''

    if dispatcher_py.exists():
        content = dispatcher_py.read_text()
        if 'sqlite3' in content and 'store_in_cortex' in content and 'INSERT INTO entries' in content:
            if DRY_RUN:
                print(f"  WOULD rewrite store_in_cortex in dispatcher.py")
            else:
                content = content.replace(old_store, new_store)
                if 'urllib.request' in content and 'tools/call' in content and 'store_in_cortex' in content:
                    dispatcher_py.write_text(content)
                    changes += 1
                    print(f"  DONE rewrite store_in_cortex → MCP API")
                else:
                    print(f"  WARN replace didn't match exactly — skipping")
        elif '127.0.0.1:8080/mcp' in content:
            print(f"  OK   store_in_cortex: already using MCP API")
        else:
            print(f"  WARN store_in_cortex: unexpected content — needs manual check")
    else:
        print(f"  SKIP dispatcher.py not found")

    # ──────────────────────────────────────────────
    # Summary
    # ──────────────────────────────────────────────
    print(f"\n{'='*60}")
    if DRY_RUN:
        print(f"DRY RUN complete. Pass --apply to make changes.")
    else:
        print(f"Applied {changes} changes.")
        print(f"\nRestart these services to pick up changes:")
        print(f"  - Dispatcher (8255)")
        print(f"  - OpenSquid daemon (8256)")
        print(f"  (Parity daemon shares port 8256 — restart whichever is active)")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
