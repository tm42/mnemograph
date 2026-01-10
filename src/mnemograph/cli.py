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

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
