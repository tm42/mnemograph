"""CLI for managing GraphMem memory outside of MCP."""

import argparse
import json
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

from .engine import MemoryEngine
from .events import EventStore
from .models import MemoryEvent
from .state import materialize


def _get_engine(args: argparse.Namespace) -> MemoryEngine:
    """Create engine instance from args."""
    memory_dir = Path(args.memory_path)
    session_id = f"cli-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}"
    return MemoryEngine(memory_dir, session_id)


def parse_since(since_str: str) -> datetime:
    """Parse relative time strings like '1 hour ago', '2 days ago'."""
    since_str = since_str.lower().strip()
    now = datetime.now(timezone.utc)

    if since_str == "today":
        return now.replace(hour=0, minute=0, second=0, microsecond=0)

    # Parse "N unit ago" format
    parts = since_str.replace(" ago", "").split()
    if len(parts) != 2:
        raise ValueError(f"Cannot parse time: {since_str}")

    amount = int(parts[0])
    unit = parts[1].rstrip("s")  # normalize: hours -> hour

    if unit == "minute":
        return now - timedelta(minutes=amount)
    elif unit == "hour":
        return now - timedelta(hours=amount)
    elif unit == "day":
        return now - timedelta(days=amount)
    elif unit == "week":
        return now - timedelta(weeks=amount)
    else:
        raise ValueError(f"Unknown time unit: {unit}")


def cmd_log(args: argparse.Namespace) -> int:
    """Show event history."""
    memory_dir = Path(args.memory_path)
    event_store = EventStore(memory_dir / "events.jsonl")
    events = event_store.read_all()

    # Filter by since
    if args.since:
        since_dt = parse_since(args.since)
        events = [e for e in events if e.ts >= since_dt]

    # Filter by session
    if args.session:
        events = [e for e in events if e.session_id == args.session]

    # Filter by operation
    if args.op:
        events = [e for e in events if e.op == args.op]

    # Apply limit (from end, most recent first)
    if args.limit:
        events = events[-args.limit:]

    # Reverse for most recent first (unless --asc)
    if not args.asc:
        events = list(reversed(events))

    if not events:
        print("No events found.")
        return 0

    # Output format
    if args.json:
        print(json.dumps([e.model_dump(mode="json") for e in events], indent=2, default=str))
    else:
        for event in events:
            ts_str = event.ts.strftime("%Y-%m-%d %H:%M:%S")
            data_summary = _summarize_event_data(event)
            print(f"{ts_str}  {event.id[:8]}  {event.op:<18}  {data_summary}")

    return 0


def _summarize_event_data(event: MemoryEvent) -> str:
    """Create a short summary of event data."""
    data = event.data
    op = event.op

    if op == "create_entity":
        return f"{data.get('name', '?')} ({data.get('type', '?')})"
    elif op == "delete_entity":
        return f"id={data.get('id', '?')[:8]}"
    elif op == "create_relation":
        return f"{data.get('from_entity', '?')[:8]} -> {data.get('to_entity', '?')[:8]}"
    elif op == "delete_relation":
        return f"{data.get('from_entity', '?')[:8]} -/-> {data.get('to_entity', '?')[:8]}"
    elif op == "add_observation":
        obs = data.get("observation", {})
        text = obs.get("text", "")[:40]
        return f"to {data.get('entity_id', '?')[:8]}: {text}..."
    elif op == "delete_observation":
        return f"from {data.get('entity_id', '?')[:8]}"
    else:
        return str(data)[:50]


def cmd_revert(args: argparse.Namespace) -> int:
    """Revert events by emitting compensating events."""
    memory_dir = Path(args.memory_path)
    event_store = EventStore(memory_dir / "events.jsonl")
    events = event_store.read_all()

    # Find events to revert
    to_revert = []

    if args.session:
        # Revert all events from a session
        to_revert = [e for e in events if e.session_id == args.session]
    elif args.event_ids:
        # Revert specific events (support full ID or 8-char prefix)
        to_revert = []
        for e in events:
            for provided_id in args.event_ids:
                if e.id == provided_id or e.id.startswith(provided_id):
                    to_revert.append(e)
                    break
    else:
        print("Error: Must specify --session or event IDs to revert.")
        return 1

    if not to_revert:
        print("No matching events found to revert.")
        return 0

    # Show what will be reverted
    print(f"Events to revert ({len(to_revert)}):")
    for event in to_revert:
        ts_str = event.ts.strftime("%Y-%m-%d %H:%M:%S")
        print(f"  {ts_str}  {event.id[:8]}  {event.op}")

    if not args.yes:
        confirm = input("\nProceed with revert? [y/N] ")
        if confirm.lower() != "y":
            print("Aborted.")
            return 0

    # Generate compensating events (reverse order)
    session_id = f"revert-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}"
    reverted = 0

    for event in reversed(to_revert):
        compensating = _create_compensating_event(event, session_id)
        if compensating:
            event_store.append(compensating)
            reverted += 1
            print(f"  Reverted: {event.op} -> {compensating.op}")

    print(f"\nReverted {reverted} events (session: {session_id})")
    return 0


def _create_compensating_event(event: MemoryEvent, session_id: str) -> MemoryEvent | None:
    """Create a compensating event to undo the given event."""
    op = event.op
    data = event.data

    if op == "create_entity":
        return MemoryEvent(
            op="delete_entity",
            session_id=session_id,
            source="user",
            data={"id": data["id"]},
        )
    elif op == "delete_entity":
        # Can't easily undo delete — would need original entity data
        # For now, skip (could store tombstones in future)
        return None
    elif op == "create_relation":
        return MemoryEvent(
            op="delete_relation",
            session_id=session_id,
            source="user",
            data={
                "from_entity": data["from_entity"],
                "to_entity": data["to_entity"],
                "type": data["type"],
            },
        )
    elif op == "delete_relation":
        # Can't easily undo — would need original relation
        return None
    elif op == "add_observation":
        return MemoryEvent(
            op="delete_observation",
            session_id=session_id,
            source="user",
            data={
                "entity_id": data["entity_id"],
                "observation_id": data["observation"]["id"],
            },
        )
    elif op == "delete_observation":
        # Can't easily undo
        return None
    else:
        return None


def cmd_status(args: argparse.Namespace) -> int:
    """Show memory status summary."""
    memory_dir = Path(args.memory_path)

    if not memory_dir.exists():
        print(f"Memory directory does not exist: {memory_dir}")
        return 1

    event_store = EventStore(memory_dir / "events.jsonl")
    events = event_store.read_all()
    state = materialize(events)

    print(f"Memory: {memory_dir}")
    print(f"Events: {len(events)}")
    print(f"Entities: {len(state.entities)}")
    print(f"Relations: {len(state.relations)}")

    # Show current branch
    engine = _get_engine(args)
    current_branch = engine.branch_manager.current_branch_name()
    print(f"Branch: {current_branch}")

    if events:
        first = events[0].ts.strftime("%Y-%m-%d %H:%M:%S")
        last = events[-1].ts.strftime("%Y-%m-%d %H:%M:%S")
        print(f"First event: {first}")
        print(f"Last event: {last}")

        # Session summary
        sessions = set(e.session_id for e in events)
        print(f"Sessions: {len(sessions)}")

    # Entity type breakdown
    if state.entities:
        types = {}
        for e in state.entities.values():
            types[e.type] = types.get(e.type, 0) + 1
        print("\nEntity types:")
        for t, count in sorted(types.items(), key=lambda x: -x[1]):
            print(f"  {t}: {count}")

    return 0


def cmd_sessions(args: argparse.Namespace) -> int:
    """List all sessions with event counts."""
    memory_dir = Path(args.memory_path)
    event_store = EventStore(memory_dir / "events.jsonl")
    events = event_store.read_all()

    if not events:
        print("No events found.")
        return 0

    # Group by session
    sessions: dict[str, list[MemoryEvent]] = {}
    for event in events:
        if event.session_id not in sessions:
            sessions[event.session_id] = []
        sessions[event.session_id].append(event)

    # Sort by first event time
    sorted_sessions = sorted(sessions.items(), key=lambda x: x[1][0].ts, reverse=True)

    print(f"{'Session ID':<40} {'Events':>8} {'First':>20} {'Last':>20}")
    print("-" * 92)

    for session_id, session_events in sorted_sessions:
        first = session_events[0].ts.strftime("%Y-%m-%d %H:%M")
        last = session_events[-1].ts.strftime("%Y-%m-%d %H:%M")
        print(f"{session_id:<40} {len(session_events):>8} {first:>20} {last:>20}")

    return 0


def cmd_session_start(args: argparse.Namespace) -> int:
    """Signal session start and get initial context."""
    memory_dir = Path(args.memory_path)

    if not memory_dir.exists():
        print(f"Memory directory does not exist: {memory_dir}")
        print("Run `mnemograph-cli status` to check your setup.")
        return 1

    session_id = args.session_id or f"cli-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}"
    engine = MemoryEngine(memory_dir, session_id)
    result = engine.session_start(project_hint=args.project)

    if args.json:
        print(json.dumps(result, indent=2, default=str))
    else:
        print(f"Session started: {result['session_id']}")
        print(f"Entities: {result['memory_summary']['entity_count']}")
        print(f"Relations: {result['memory_summary']['relation_count']}")
        if result.get('project'):
            print(f"Project: {result['project']}")
        print()
        print("Context:")
        print(result['context'])

    return 0


def cmd_session_end(args: argparse.Namespace) -> int:
    """Signal session end, optionally save summary."""
    memory_dir = Path(args.memory_path)

    if not memory_dir.exists():
        print(f"Memory directory does not exist: {memory_dir}")
        return 1

    session_id = args.session_id or f"cli-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}"
    engine = MemoryEngine(memory_dir, session_id)
    result = engine.session_end(summary=args.summary)

    if args.json:
        print(json.dumps(result, indent=2, default=str))
    else:
        print(f"Session ended: {result['status']}")
        if result.get('summary_stored'):
            print("Summary stored in knowledge graph.")
        print(f"Tip: {result['tip']}")

    return 0


def cmd_primer(args: argparse.Namespace) -> int:
    """Get orientation primer for the knowledge graph."""
    memory_dir = Path(args.memory_path)

    if not memory_dir.exists():
        print(f"Memory directory does not exist: {memory_dir}")
        return 1

    session_id = f"cli-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}"
    engine = MemoryEngine(memory_dir, session_id)
    result = engine.get_primer()

    if args.json:
        print(json.dumps(result, indent=2, default=str))
    else:
        print("=== Knowledge Graph Primer ===\n")
        print(f"Entities: {result['status']['entity_count']}")
        print(f"Relations: {result['status']['relation_count']}")

        if result['status']['types']:
            print("\nEntity types:")
            for t, count in result['status']['types'].items():
                print(f"  {t}: {count}")

        if result['recent_activity']:
            print("\nRecent activity:")
            for e in result['recent_activity']:
                print(f"  - {e['name']} ({e['type']})")

        print(f"\nQuick start: {result['quick_start']}")

    return 0


def cmd_export(args: argparse.Namespace) -> int:
    """Export graph as JSON."""
    memory_dir = Path(args.memory_path)

    if not memory_dir.exists():
        print(f"Memory directory does not exist: {memory_dir}", file=sys.stderr)
        return 1

    event_store = EventStore(memory_dir / "events.jsonl")
    events = event_store.read_all()
    state = materialize(events)

    result = {
        "entities": [e.model_dump(mode="json") for e in state.entities.values()],
        "relations": [r.model_dump(mode="json") for r in state.relations],
    }

    output = json.dumps(result, indent=2, default=str)

    if args.output:
        Path(args.output).write_text(output)
        print(f"Exported to {args.output}", file=sys.stderr)
    else:
        print(output)

    return 0


# ─────────────────────────────────────────────────────────────────────────────
# Branch Commands
# ─────────────────────────────────────────────────────────────────────────────


def cmd_branch_list(args: argparse.Namespace) -> int:
    """List all branches."""
    memory_dir = Path(args.memory_path)

    if not memory_dir.exists():
        print(f"Memory directory does not exist: {memory_dir}", file=sys.stderr)
        return 1

    engine = _get_engine(args)
    branches = engine.branch_manager.list(include_archived=args.all)
    current = engine.branch_manager.current_branch_name()

    if args.json:
        result = [
            {
                "name": b.name,
                "is_current": b.name == current,
                "entity_count": len(b.entity_ids) if b.name != "main" else len(engine.state.entities),
                "relation_count": len(b.relation_ids) if b.name != "main" else len(engine.state.relations),
                "description": b.description,
                "is_active": b.is_active,
            }
            for b in branches
        ]
        print(json.dumps(result, indent=2))
    else:
        for branch in branches:
            marker = "*" if branch.name == current else " "
            if branch.name == "main":
                entity_count = len(engine.state.entities)
                rel_count = len(engine.state.relations)
            else:
                entity_count = len(branch.entity_ids)
                rel_count = len(branch.relation_ids)
            status = "" if branch.is_active else " [archived]"
            print(f"{marker} {branch.name:<30} ({entity_count} entities, {rel_count} relations){status}")

    return 0


def cmd_branch_show(args: argparse.Namespace) -> int:
    """Show details of a branch."""
    memory_dir = Path(args.memory_path)

    if not memory_dir.exists():
        print(f"Memory directory does not exist: {memory_dir}", file=sys.stderr)
        return 1

    engine = _get_engine(args)
    branch_name = args.name or engine.branch_manager.current_branch_name()

    try:
        branch = engine.branch_manager.get(branch_name)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    if args.json:
        result = branch.to_dict()
        result["is_current"] = branch_name == engine.branch_manager.current_branch_name()
        print(json.dumps(result, indent=2, default=str))
    else:
        current = engine.branch_manager.current_branch_name()
        marker = " (current)" if branch_name == current else ""
        print(f"Branch: {branch.name}{marker}")
        print(f"Description: {branch.description or '(none)'}")
        print(f"Parent: {branch.parent or '(none)'}")
        print(f"Created: {branch.created_at.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Active: {branch.is_active}")

        if branch.name == "main":
            print(f"Entities: {len(engine.state.entities)} (all)")
            print(f"Relations: {len(engine.state.relations)} (all)")
        else:
            print(f"Entities: {len(branch.entity_ids)}")
            print(f"Relations: {len(branch.relation_ids)}")

            if args.verbose and branch.entity_ids:
                print("\nEntity IDs:")
                for eid in sorted(branch.entity_ids)[:20]:
                    entity = engine.state.entities.get(eid)
                    name = entity.name if entity else "(deleted)"
                    print(f"  {eid[:8]}  {name}")
                if len(branch.entity_ids) > 20:
                    print(f"  ... and {len(branch.entity_ids) - 20} more")

    return 0


def cmd_branch_create(args: argparse.Namespace) -> int:
    """Create a new branch."""
    memory_dir = Path(args.memory_path)

    if not memory_dir.exists():
        print(f"Memory directory does not exist: {memory_dir}", file=sys.stderr)
        return 1

    engine = _get_engine(args)

    # Parse seed entities
    seeds = args.seeds if args.seeds else []

    try:
        branch = engine.branch_manager.create(
            name=args.name,
            seed_entities=seeds,
            description=args.description or "",
            depth=args.depth,
        )
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    if args.checkout:
        engine.branch_manager.checkout(args.name)
        print(f"Created and switched to branch '{args.name}'")
    else:
        print(f"Created branch '{args.name}'")

    print(f"  Entities: {len(branch.entity_ids)}")
    print(f"  Relations: {len(branch.relation_ids)}")

    return 0


def cmd_branch_delete(args: argparse.Namespace) -> int:
    """Delete a branch."""
    memory_dir = Path(args.memory_path)

    if not memory_dir.exists():
        print(f"Memory directory does not exist: {memory_dir}", file=sys.stderr)
        return 1

    engine = _get_engine(args)

    if not args.yes:
        confirm = input(f"Delete branch '{args.name}'? [y/N] ")
        if confirm.lower() != "y":
            print("Aborted.")
            return 0

    try:
        engine.branch_manager.delete(args.name)
        print(f"Deleted branch '{args.name}'")
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    return 0


def cmd_branch_checkout(args: argparse.Namespace) -> int:
    """Switch to a branch."""
    memory_dir = Path(args.memory_path)

    if not memory_dir.exists():
        print(f"Memory directory does not exist: {memory_dir}", file=sys.stderr)
        return 1

    engine = _get_engine(args)

    try:
        branch = engine.branch_manager.checkout(args.name)
        print(f"Switched to branch '{args.name}'")
        if branch.name != "main":
            print(f"  Entities: {len(branch.entity_ids)}")
            print(f"  Relations: {len(branch.relation_ids)}")
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    return 0


def cmd_branch_add(args: argparse.Namespace) -> int:
    """Add entities to the current branch."""
    memory_dir = Path(args.memory_path)

    if not memory_dir.exists():
        print(f"Memory directory does not exist: {memory_dir}", file=sys.stderr)
        return 1

    engine = _get_engine(args)
    current = engine.branch_manager.current_branch_name()

    if current == "main":
        print("Cannot add entities to main branch (main sees everything).", file=sys.stderr)
        return 1

    # Resolve entity names to IDs
    entity_ids = []
    for name in args.entities:
        eid = engine._resolve_entity(name)
        if eid:
            entity_ids.append(eid)
        else:
            print(f"Warning: Entity not found: {name}", file=sys.stderr)

    if not entity_ids:
        print("No valid entities to add.", file=sys.stderr)
        return 1

    try:
        branch = engine.branch_manager.add_entities(
            current,
            entity_ids,
            include_relations=not args.no_relations,
        )
        print(f"Added {len(entity_ids)} entities to '{current}'")
        print(f"  Total entities: {len(branch.entity_ids)}")
        print(f"  Total relations: {len(branch.relation_ids)}")
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    return 0


def cmd_branch_remove(args: argparse.Namespace) -> int:
    """Remove entities from the current branch."""
    memory_dir = Path(args.memory_path)

    if not memory_dir.exists():
        print(f"Memory directory does not exist: {memory_dir}", file=sys.stderr)
        return 1

    engine = _get_engine(args)
    current = engine.branch_manager.current_branch_name()

    if current == "main":
        print("Cannot remove entities from main branch.", file=sys.stderr)
        return 1

    # Resolve entity names to IDs
    entity_ids = []
    for name in args.entities:
        eid = engine._resolve_entity(name)
        if eid:
            entity_ids.append(eid)
        else:
            print(f"Warning: Entity not found: {name}", file=sys.stderr)

    if not entity_ids:
        print("No valid entities to remove.", file=sys.stderr)
        return 1

    try:
        branch = engine.branch_manager.remove_entities(
            current,
            entity_ids,
            cascade_relations=not args.keep_relations,
        )
        print(f"Removed {len(entity_ids)} entities from '{current}'")
        print(f"  Remaining entities: {len(branch.entity_ids)}")
        print(f"  Remaining relations: {len(branch.relation_ids)}")
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    return 0


def cmd_branch_diff(args: argparse.Namespace) -> int:
    """Show diff between two branches."""
    memory_dir = Path(args.memory_path)

    if not memory_dir.exists():
        print(f"Memory directory does not exist: {memory_dir}", file=sys.stderr)
        return 1

    engine = _get_engine(args)

    branch_a = args.branch_a or engine.branch_manager.current_branch_name()
    branch_b = args.branch_b

    try:
        diff = engine.branch_manager.diff(branch_a, branch_b)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    if args.json:
        print(json.dumps(diff, indent=2))
    else:
        print(f"Diff: {diff['branch_a']} vs {diff['branch_b']}")
        print()

        # Only in A
        if diff["only_in_a"]["entities"]:
            print(f"Only in {diff['branch_a']} ({len(diff['only_in_a']['entities'])} entities):")
            for eid in list(diff["only_in_a"]["entities"])[:10]:
                entity = engine.state.entities.get(eid)
                name = entity.name if entity else eid[:8]
                print(f"  - {name}")
            if len(diff["only_in_a"]["entities"]) > 10:
                print(f"  ... and {len(diff['only_in_a']['entities']) - 10} more")
            print()

        # Only in B
        if diff["only_in_b"]["entities"]:
            print(f"Only in {diff['branch_b']} ({len(diff['only_in_b']['entities'])} entities):")
            for eid in list(diff["only_in_b"]["entities"])[:10]:
                entity = engine.state.entities.get(eid)
                name = entity.name if entity else eid[:8]
                print(f"  + {name}")
            if len(diff["only_in_b"]["entities"]) > 10:
                print(f"  ... and {len(diff['only_in_b']['entities']) - 10} more")
            print()

        # In both
        if diff["in_both"]["entities"]:
            print(f"In both ({len(diff['in_both']['entities'])} entities)")

    return 0


def cmd_branch_merge(args: argparse.Namespace) -> int:
    """Merge source branch into target."""
    memory_dir = Path(args.memory_path)

    if not memory_dir.exists():
        print(f"Memory directory does not exist: {memory_dir}", file=sys.stderr)
        return 1

    engine = _get_engine(args)
    target = args.target or engine.branch_manager.current_branch_name()

    try:
        branch = engine.branch_manager.merge(args.source, target)
        print(f"Merged '{args.source}' into '{target}'")
        print(f"  Entities: {len(branch.entity_ids)}")
        print(f"  Relations: {len(branch.relation_ids)}")
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    return 0


def cmd_branch_archive(args: argparse.Namespace) -> int:
    """Archive a branch."""
    memory_dir = Path(args.memory_path)

    if not memory_dir.exists():
        print(f"Memory directory does not exist: {memory_dir}", file=sys.stderr)
        return 1

    engine = _get_engine(args)

    try:
        branch = engine.branch_manager.archive(args.name)
        print(f"Archived branch '{args.name}' -> '{branch.name}'")
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    return 0


def cmd_branch_unarchive(args: argparse.Namespace) -> int:
    """Unarchive a branch."""
    memory_dir = Path(args.memory_path)

    if not memory_dir.exists():
        print(f"Memory directory does not exist: {memory_dir}", file=sys.stderr)
        return 1

    engine = _get_engine(args)

    try:
        branch = engine.branch_manager.unarchive(args.name, new_name=args.new_name)
        print(f"Unarchived branch '{args.name}' -> '{branch.name}'")
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    return 0


def main(argv: list[str] | None = None) -> int:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="mnemograph-cli",
        description="Manage Mnemograph memory system",
    )
    parser.add_argument(
        "--memory-path",
        default=os.environ.get("MEMORY_PATH", ".claude/memory"),
        help="Path to memory directory (default: $MEMORY_PATH or .claude/memory)",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    # log command
    log_parser = subparsers.add_parser("log", help="Show event history")
    log_parser.add_argument("--since", help="Show events since (e.g., '1 hour ago', 'today')")
    log_parser.add_argument("--session", help="Filter by session ID")
    log_parser.add_argument("--op", help="Filter by operation type")
    log_parser.add_argument("--limit", "-n", type=int, help="Limit number of events")
    log_parser.add_argument("--asc", action="store_true", help="Show oldest first")
    log_parser.add_argument("--json", action="store_true", help="Output as JSON")
    log_parser.set_defaults(func=cmd_log)

    # revert command
    revert_parser = subparsers.add_parser("revert", help="Revert events")
    revert_parser.add_argument("--session", help="Revert all events from session")
    revert_parser.add_argument("event_ids", nargs="*", help="Specific event IDs to revert")
    revert_parser.add_argument("-y", "--yes", action="store_true", help="Skip confirmation")
    revert_parser.set_defaults(func=cmd_revert)

    # status command
    status_parser = subparsers.add_parser("status", help="Show memory status")
    status_parser.set_defaults(func=cmd_status)

    # sessions command
    sessions_parser = subparsers.add_parser("sessions", help="List all sessions")
    sessions_parser.set_defaults(func=cmd_sessions)

    # session start command
    session_start_parser = subparsers.add_parser(
        "session-start",
        help="Signal session start and get initial context",
    )
    session_start_parser.add_argument(
        "--project", help="Project name or path for context"
    )
    session_start_parser.add_argument(
        "--session-id", help="Custom session ID (default: auto-generated)"
    )
    session_start_parser.add_argument(
        "--json", action="store_true", help="Output as JSON"
    )
    session_start_parser.set_defaults(func=cmd_session_start)

    # session end command
    session_end_parser = subparsers.add_parser(
        "session-end",
        help="Signal session end, optionally save summary",
    )
    session_end_parser.add_argument(
        "--summary", "-m", help="Session summary to store"
    )
    session_end_parser.add_argument(
        "--session-id", help="Custom session ID (default: auto-generated)"
    )
    session_end_parser.add_argument(
        "--json", action="store_true", help="Output as JSON"
    )
    session_end_parser.set_defaults(func=cmd_session_end)

    # primer command
    primer_parser = subparsers.add_parser(
        "primer",
        help="Get orientation primer for the knowledge graph",
    )
    primer_parser.add_argument(
        "--json", action="store_true", help="Output as JSON"
    )
    primer_parser.set_defaults(func=cmd_primer)

    # export command (existing but let's verify it's here or add it)
    export_parser = subparsers.add_parser("export", help="Export graph as JSON")
    export_parser.add_argument(
        "--output", "-o", help="Output file (default: stdout)"
    )
    export_parser.set_defaults(func=cmd_export)

    # ─────────────────────────────────────────────────────────────────────────
    # Branch commands
    # ─────────────────────────────────────────────────────────────────────────

    branch_parser = subparsers.add_parser("branch", help="Branch management commands")
    branch_subparsers = branch_parser.add_subparsers(dest="branch_command", required=True)

    # branch list
    branch_list_parser = branch_subparsers.add_parser("list", help="List all branches")
    branch_list_parser.add_argument("--all", "-a", action="store_true", help="Include archived branches")
    branch_list_parser.add_argument("--json", action="store_true", help="Output as JSON")
    branch_list_parser.set_defaults(func=cmd_branch_list)

    # branch show
    branch_show_parser = branch_subparsers.add_parser("show", help="Show branch details")
    branch_show_parser.add_argument("name", nargs="?", help="Branch name (default: current)")
    branch_show_parser.add_argument("--verbose", "-v", action="store_true", help="Show entity details")
    branch_show_parser.add_argument("--json", action="store_true", help="Output as JSON")
    branch_show_parser.set_defaults(func=cmd_branch_show)

    # branch create
    branch_create_parser = branch_subparsers.add_parser("create", help="Create a new branch")
    branch_create_parser.add_argument("name", help="Branch name (e.g., project/my-project)")
    branch_create_parser.add_argument("--seeds", "-s", nargs="*", help="Seed entity names")
    branch_create_parser.add_argument("--depth", "-d", type=int, default=2, help="Expansion depth (default: 2)")
    branch_create_parser.add_argument("--description", "-m", help="Branch description")
    branch_create_parser.add_argument("--checkout", "-c", action="store_true", help="Switch to new branch")
    branch_create_parser.set_defaults(func=cmd_branch_create)

    # branch delete
    branch_delete_parser = branch_subparsers.add_parser("delete", help="Delete a branch")
    branch_delete_parser.add_argument("name", help="Branch name to delete")
    branch_delete_parser.add_argument("-y", "--yes", action="store_true", help="Skip confirmation")
    branch_delete_parser.set_defaults(func=cmd_branch_delete)

    # branch checkout (also add as top-level 'checkout' command)
    branch_checkout_parser = branch_subparsers.add_parser("checkout", help="Switch to a branch")
    branch_checkout_parser.add_argument("name", help="Branch name to switch to")
    branch_checkout_parser.set_defaults(func=cmd_branch_checkout)

    # top-level checkout shortcut
    checkout_parser = subparsers.add_parser("checkout", help="Switch to a branch (shortcut)")
    checkout_parser.add_argument("name", help="Branch name to switch to")
    checkout_parser.set_defaults(func=cmd_branch_checkout)

    # branch add
    branch_add_parser = branch_subparsers.add_parser("add", help="Add entities to current branch")
    branch_add_parser.add_argument("entities", nargs="+", help="Entity names to add")
    branch_add_parser.add_argument("--no-relations", action="store_true", help="Don't include relations")
    branch_add_parser.set_defaults(func=cmd_branch_add)

    # branch remove
    branch_remove_parser = branch_subparsers.add_parser("remove", help="Remove entities from current branch")
    branch_remove_parser.add_argument("entities", nargs="+", help="Entity names to remove")
    branch_remove_parser.add_argument("--keep-relations", action="store_true", help="Don't cascade relation removal")
    branch_remove_parser.set_defaults(func=cmd_branch_remove)

    # branch diff
    branch_diff_parser = branch_subparsers.add_parser("diff", help="Show diff between branches")
    branch_diff_parser.add_argument("branch_a", nargs="?", help="First branch (default: current)")
    branch_diff_parser.add_argument("branch_b", help="Second branch to compare")
    branch_diff_parser.add_argument("--json", action="store_true", help="Output as JSON")
    branch_diff_parser.set_defaults(func=cmd_branch_diff)

    # branch merge
    branch_merge_parser = branch_subparsers.add_parser("merge", help="Merge source branch into target")
    branch_merge_parser.add_argument("source", help="Source branch to merge from")
    branch_merge_parser.add_argument("--target", "-t", help="Target branch (default: current)")
    branch_merge_parser.set_defaults(func=cmd_branch_merge)

    # branch archive
    branch_archive_parser = branch_subparsers.add_parser("archive", help="Archive a branch")
    branch_archive_parser.add_argument("name", help="Branch name to archive")
    branch_archive_parser.set_defaults(func=cmd_branch_archive)

    # branch unarchive
    branch_unarchive_parser = branch_subparsers.add_parser("unarchive", help="Unarchive a branch")
    branch_unarchive_parser.add_argument("name", help="Archived branch name")
    branch_unarchive_parser.add_argument("--new-name", help="New name for restored branch")
    branch_unarchive_parser.set_defaults(func=cmd_branch_unarchive)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
