"""Unified CLI for Mnemograph memory management.

Combines memory operations and version control in a single interface.
"""

import json
import os
from datetime import datetime, timezone
from pathlib import Path

import click
from rich.console import Console
from rich.table import Table

from .engine import MemoryEngine
from .events import EventStore
from .models import MemoryEvent
from .state import materialize
from .timeutil import parse_time_reference, format_relative_time
from .vcs import MemoryVCS

console = Console()


# ─────────────────────────────────────────────────────────────────────────────
# Shared Helpers
# ─────────────────────────────────────────────────────────────────────────────


def get_memory_dir() -> Path:
    """Find memory directory from MEMORY_PATH env or walk up to find .claude/memory."""
    if env_path := os.environ.get("MEMORY_PATH"):
        return Path(env_path)

    cwd = Path.cwd()
    for parent in [cwd] + list(cwd.parents):
        memory_dir = parent / ".claude" / "memory"
        if memory_dir.exists():
            return memory_dir

    return cwd / ".claude" / "memory"


def get_engine(memory_dir: Path) -> MemoryEngine:
    """Create engine instance for CLI operations."""
    session_id = f"cli-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}"
    return MemoryEngine(memory_dir, session_id)


def summarize_event_data(event: MemoryEvent) -> str:
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


def parse_since(since_str: str) -> datetime:
    """Parse relative time strings like '1 hour ago', '2 days ago'."""
    from datetime import timedelta
    since_str = since_str.lower().strip()
    now = datetime.now(timezone.utc)

    if since_str == "today":
        return now.replace(hour=0, minute=0, second=0, microsecond=0)

    parts = since_str.replace(" ago", "").split()
    if len(parts) != 2:
        raise ValueError(f"Cannot parse time: {since_str}")

    amount = int(parts[0])
    unit = parts[1].rstrip("s")

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


# ─────────────────────────────────────────────────────────────────────────────
# Main CLI Group
# ─────────────────────────────────────────────────────────────────────────────


@click.group()
@click.option(
    "--memory-path",
    envvar="MEMORY_PATH",
    type=click.Path(path_type=Path),
    help="Path to memory directory",
)
@click.option(
    "--global", "use_global",
    is_flag=True,
    help="Use global memory (~/.claude/memory) instead of project-local",
)
@click.pass_context
def cli(ctx, memory_path, use_global):
    """Mnemograph - Knowledge graph memory for AI agents."""
    ctx.ensure_object(dict)
    if use_global:
        ctx.obj["memory_dir"] = Path.home() / ".claude" / "memory"
    else:
        ctx.obj["memory_dir"] = memory_path or get_memory_dir()


# ─────────────────────────────────────────────────────────────────────────────
# Core Commands
# ─────────────────────────────────────────────────────────────────────────────


@cli.command()
@click.pass_context
def status(ctx):
    """Show memory and repository status."""
    memory_dir = ctx.obj["memory_dir"]

    if not memory_dir.exists():
        console.print(f"[red]Memory directory does not exist:[/red] {memory_dir}")
        return

    # Load state
    event_store = EventStore(memory_dir / "mnemograph.db")
    events = event_store.read_all()
    state = materialize(events)

    console.print(f"Memory: [cyan]{memory_dir}[/cyan]")
    console.print(f"Events: {len(events)}")
    console.print(f"Entities: [bold]{len(state.entities)}[/bold]")
    console.print(f"Relations: [bold]{len(state.relations)}[/bold]")

    # Show current branch
    engine = get_engine(memory_dir)
    current_branch = engine.branch_manager.current_branch_name()
    console.print(f"Branch: [cyan]{current_branch}[/cyan]")

    # VCS status if available
    try:
        vcs = MemoryVCS(memory_dir)
        vcs_status = vcs.status()
        if vcs_status.get("commit"):
            console.print(f"Last commit: [yellow]{vcs_status['commit']}[/yellow] {vcs_status.get('commit_message', '')}")
        if vcs_status.get("is_dirty"):
            console.print(f"[yellow]Uncommitted events: {vcs_status.get('event_count', '?')}[/yellow]")
    except RuntimeError:
        pass  # Not a git repo

    if events:
        first = events[0].ts.strftime("%Y-%m-%d %H:%M:%S")
        last = events[-1].ts.strftime("%Y-%m-%d %H:%M:%S")
        console.print(f"First event: {first}")
        console.print(f"Last event: {last}")

        sessions = set(e.session_id for e in events)
        console.print(f"Sessions: {len(sessions)}")

    if state.entities:
        types = {}
        for e in state.entities.values():
            types[e.type] = types.get(e.type, 0) + 1
        console.print("\nEntity types:")
        for t, count in sorted(types.items(), key=lambda x: -x[1]):
            console.print(f"  {t}: {count}")


@cli.command("log")
@click.option("--since", help="Show events since (e.g., '1 hour ago', 'today')")
@click.option("--session", help="Filter by session ID")
@click.option("--op", help="Filter by operation type")
@click.option("-n", "--limit", type=int, help="Limit number of events")
@click.option("--asc", is_flag=True, help="Show oldest first")
@click.option("--json", "as_json", is_flag=True, help="Output as JSON")
@click.pass_context
def event_log(ctx, since, session, op, limit, asc, as_json):
    """Show event history."""
    memory_dir = ctx.obj["memory_dir"]
    event_store = EventStore(memory_dir / "mnemograph.db")
    events = event_store.read_all()

    if since:
        since_dt = parse_since(since)
        events = [e for e in events if e.ts >= since_dt]

    if session:
        events = [e for e in events if e.session_id == session]

    if op:
        events = [e for e in events if e.op == op]

    if limit:
        events = events[-limit:]

    if not asc:
        events = list(reversed(events))

    if not events:
        console.print("No events found.")
        return

    if as_json:
        console.print(json.dumps([e.model_dump(mode="json") for e in events], indent=2, default=str))
    else:
        for event in events:
            ts_str = event.ts.strftime("%Y-%m-%d %H:%M:%S")
            data_summary = summarize_event_data(event)
            console.print(f"{ts_str}  {event.id[:8]}  {event.op:<18}  {data_summary}")


@cli.command()
@click.pass_context
def sessions(ctx):
    """List all sessions with event counts."""
    memory_dir = ctx.obj["memory_dir"]
    event_store = EventStore(memory_dir / "mnemograph.db")
    events = event_store.read_all()

    if not events:
        console.print("No events found.")
        return

    session_map: dict[str, list[MemoryEvent]] = {}
    for event in events:
        if event.session_id not in session_map:
            session_map[event.session_id] = []
        session_map[event.session_id].append(event)

    sorted_sessions = sorted(session_map.items(), key=lambda x: x[1][0].ts, reverse=True)

    console.print(f"{'Session ID':<40} {'Events':>8} {'First':>20} {'Last':>20}")
    console.print("-" * 92)

    for session_id, session_events in sorted_sessions:
        first = session_events[0].ts.strftime("%Y-%m-%d %H:%M")
        last = session_events[-1].ts.strftime("%Y-%m-%d %H:%M")
        console.print(f"{session_id:<40} {len(session_events):>8} {first:>20} {last:>20}")


@cli.command()
@click.option("--output", "-o", type=click.Path(path_type=Path), help="Output file (default: stdout)")
@click.pass_context
def export(ctx, output):
    """Export graph as JSON."""
    memory_dir = ctx.obj["memory_dir"]

    if not memory_dir.exists():
        console.print(f"[red]Memory directory does not exist:[/red] {memory_dir}")
        return

    event_store = EventStore(memory_dir / "mnemograph.db")
    events = event_store.read_all()
    state = materialize(events)

    result = {
        "entities": [e.model_dump(mode="json") for e in state.entities.values()],
        "relations": [r.model_dump(mode="json") for r in state.relations],
    }

    output_str = json.dumps(result, indent=2, default=str)

    if output:
        output.write_text(output_str)
        console.print(f"[green]✓[/green] Exported to {output}")
    else:
        console.print(output_str)


@cli.command("session-start")
@click.option("--project", help="Project name or path for context")
@click.option("--session-id", help="Custom session ID")
@click.option("--json", "as_json", is_flag=True, help="Output as JSON")
@click.pass_context
def session_start(ctx, project, session_id, as_json):
    """Signal session start and get initial context."""
    memory_dir = ctx.obj["memory_dir"]

    if not memory_dir.exists():
        console.print(f"[red]Memory directory does not exist:[/red] {memory_dir}")
        return

    sid = session_id or f"cli-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}"
    engine = MemoryEngine(memory_dir, sid)
    result = engine.session_start(project_hint=project)

    if as_json:
        console.print(json.dumps(result, indent=2, default=str))
    else:
        console.print(f"Session started: {result['session_id']}")
        console.print(f"Entities: {result['memory_summary']['entity_count']}")
        console.print(f"Relations: {result['memory_summary']['relation_count']}")
        if result.get('project'):
            console.print(f"Project: {result['project']}")
        console.print()
        console.print("Context:")
        console.print(result['context'])


@cli.command("session-end")
@click.option("--summary", "-m", help="Session summary to store")
@click.option("--session-id", help="Custom session ID")
@click.option("--json", "as_json", is_flag=True, help="Output as JSON")
@click.pass_context
def session_end(ctx, summary, session_id, as_json):
    """Signal session end, optionally save summary."""
    memory_dir = ctx.obj["memory_dir"]

    if not memory_dir.exists():
        console.print(f"[red]Memory directory does not exist:[/red] {memory_dir}")
        return

    sid = session_id or f"cli-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}"
    engine = MemoryEngine(memory_dir, sid)
    result = engine.session_end(summary=summary)

    if as_json:
        console.print(json.dumps(result, indent=2, default=str))
    else:
        console.print(f"Session ended: {result['status']}")
        if result.get('summary_stored'):
            console.print("Summary stored in knowledge graph.")
        console.print(f"Tip: {result['tip']}")


@cli.command()
@click.option("--json", "as_json", is_flag=True, help="Output as JSON")
@click.pass_context
def primer(ctx, as_json):
    """Get orientation primer for the knowledge graph."""
    memory_dir = ctx.obj["memory_dir"]

    if not memory_dir.exists():
        console.print(f"[red]Memory directory does not exist:[/red] {memory_dir}")
        return

    engine = get_engine(memory_dir)
    result = engine.get_primer()

    if as_json:
        console.print(json.dumps(result, indent=2, default=str))
    else:
        console.print("=== Knowledge Graph Primer ===\n")
        console.print(f"Entities: {result['status']['entity_count']}")
        console.print(f"Relations: {result['status']['relation_count']}")

        if result['status']['types']:
            console.print("\nEntity types:")
            for t, count in result['status']['types'].items():
                console.print(f"  {t}: {count}")

        if result['recent_activity']:
            console.print("\nRecent activity:")
            for e in result['recent_activity']:
                console.print(f"  - {e['name']} ({e['type']})")

        console.print(f"\nQuick start: {result['quick_start']}")


# ─────────────────────────────────────────────────────────────────────────────
# VCS Commands (git-based versioning)
# ─────────────────────────────────────────────────────────────────────────────


@cli.group()
def vcs():
    """Version control commands (git-based)."""
    pass


@vcs.command()
@click.pass_context
def init(ctx):
    """Initialize memory repository."""
    vcs_obj = MemoryVCS(ctx.obj["memory_dir"])
    if vcs_obj.init():
        console.print(f"[green]✓[/green] Initialized memory repository at {ctx.obj['memory_dir']}")
    else:
        console.print(f"[yellow]![/yellow] Repository already initialized at {ctx.obj['memory_dir']}")


@vcs.command()
@click.option("-m", "--message", required=True, help="Commit message")
@click.option("-a", "--auto-summary", is_flag=True, help="Add auto-generated summary")
@click.pass_context
def commit(ctx, message, auto_summary):
    """Commit changes to events."""
    try:
        vcs_obj = MemoryVCS(ctx.obj["memory_dir"])
        commit_hash = vcs_obj.commit(message, auto_summary=auto_summary)
        console.print(f"[green]✓[/green] [{commit_hash}] {message}")
    except RuntimeError as e:
        console.print(f"[red]Error:[/red] {e}")


@vcs.command("log")
@click.option("-n", "--max-count", default=10, help="Number of commits to show")
@click.option("--oneline", is_flag=True, help="Show one line per commit")
@click.pass_context
def vcs_log(ctx, max_count, oneline):
    """Show commit logs."""
    try:
        vcs_obj = MemoryVCS(ctx.obj["memory_dir"])
        commits = vcs_obj.log(n=max_count)
    except RuntimeError as e:
        console.print(f"[red]Error:[/red] {e}")
        return

    for c in commits:
        if oneline:
            console.print(
                f"[yellow]{c['short_hash']}[/yellow] "
                f"{c['message'].split(chr(10))[0]}"
            )
        else:
            console.print(f"[yellow]commit {c['hash']}[/yellow]")
            console.print(f"Author: {c['author']}")
            console.print(f"Date:   {c['date']}")
            console.print()
            for line in c["message"].split("\n"):
                console.print(f"    {line}")
            console.print()
            stats = c["stats"]
            console.print(f"    [dim]Entities: {stats['entities']}, Relations: {stats['relations']}[/dim]")
            console.print()


@vcs.command()
@click.option("--session", "-s", help="Revert all events from this session")
@click.option("--event", "-e", "event_ids", multiple=True, help="Revert specific event IDs (can specify multiple)")
@click.option("--yes", "-y", is_flag=True, help="Skip confirmation prompt")
@click.pass_context
def revert(ctx, session, event_ids, yes):
    """Revert events by emitting compensating events."""
    memory_dir = ctx.obj["memory_dir"]
    event_store = EventStore(memory_dir / "mnemograph.db")
    events = event_store.read_all()

    # Find events to revert
    to_revert = []

    if session:
        # Revert all events from a session
        to_revert = [e for e in events if e.session_id == session]
    elif event_ids:
        # Revert specific events (support full ID or 8-char prefix)
        to_revert = []
        for e in events:
            for provided_id in event_ids:
                if e.id == provided_id or e.id.startswith(provided_id):
                    to_revert.append(e)
                    break
    else:
        console.print("[red]Error:[/red] Must specify --session or --event to revert.")
        raise SystemExit(1)

    if not to_revert:
        console.print("[yellow]No matching events found to revert.[/yellow]")
        return

    # Show what will be reverted
    console.print(f"[bold]Events to revert ({len(to_revert)}):[/bold]")
    for event in to_revert:
        ts_str = event.ts.strftime("%Y-%m-%d %H:%M:%S") if hasattr(event.ts, 'strftime') else str(event.ts)[:19]
        console.print(f"  {ts_str}  {event.id[:8]}  {event.op}")

    if not yes:
        if not click.confirm("\nProceed with revert?", default=False):
            console.print("[dim]Cancelled.[/dim]")
            return

    # Generate compensating events (reverse order)
    revert_session_id = f"revert-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}"
    reverted = 0

    for event in reversed(to_revert):
        compensating = _create_compensating_event(event, revert_session_id)
        if compensating:
            event_store.append(compensating)
            reverted += 1
            console.print(f"  [green]✓[/green] Reverted: {event.op} → {compensating.op}")

    console.print(f"\n[green]Reverted {reverted} events[/green] (session: {revert_session_id})")


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
        # Can't easily undo — would need original observation
        return None
    else:
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Time Travel Commands
# ─────────────────────────────────────────────────────────────────────────────


@cli.command()
@click.option("--at", "timestamp", help="Show state at this time (ISO, relative, or named)")
@click.option("--ago", help="Show state N units ago (e.g., '7 days', '2 weeks')")
@click.option("--json", "as_json", is_flag=True, help="Output as JSON")
@click.pass_context
def show(ctx, timestamp, ago, as_json):
    """Show graph state at a point in time."""
    memory_dir = ctx.obj["memory_dir"]

    if ago:
        ts_str = f"{ago} ago"
    elif timestamp:
        ts_str = timestamp
    else:
        ts_str = None

    try:
        engine = get_engine(memory_dir)

        if ts_str:
            ts = parse_time_reference(ts_str)
            state = engine.state_at(ts)
            header = f"State at {ts.strftime('%Y-%m-%d %H:%M:%S')} ({format_relative_time(ts)})"
        else:
            state = engine.state
            header = "Current state"

        if as_json:
            result = {
                "timestamp": ts.isoformat() if ts_str else datetime.now(timezone.utc).isoformat(),
                "entity_count": len(state.entities),
                "relation_count": len(state.relations),
                "entities": [e.to_summary() for e in state.entities.values()],
            }
            console.print(json.dumps(result, indent=2))
        else:
            console.print(f"[bold]{header}[/bold]")
            console.print(f"Entities: {len(state.entities)}, Relations: {len(state.relations)}")
            console.print()

            if state.entities:
                table = Table(title="Entities")
                table.add_column("Name", style="cyan")
                table.add_column("Type", style="green")
                table.add_column("Observations")

                for entity in sorted(state.entities.values(), key=lambda e: e.name):
                    table.add_row(entity.name, entity.type, str(len(entity.observations)))

                console.print(table)
            else:
                console.print("[dim]No entities[/dim]")

    except ValueError as e:
        console.print(f"[red]Error:[/red] {e}")


@cli.command()
@click.argument("start")
@click.option("--to", "end", default=None, help="End time (default: now)")
@click.option("--json", "as_json", is_flag=True, help="Output as JSON")
@click.pass_context
def diff(ctx, start, end, as_json):
    """Show what changed between two points in time."""
    memory_dir = ctx.obj["memory_dir"]

    try:
        engine = get_engine(memory_dir)
        result = engine.diff_between(start, end)

        if as_json:
            console.print(json.dumps(result, indent=2))
        else:
            console.print(f"[bold]Changes from {result['start_timestamp']} to {result['end_timestamp']}[/bold]")
            console.print(f"Events in range: {result['event_count']}")
            console.print()

            entities = result["entities"]
            relations = result["relations"]

            if entities["added"]:
                console.print("[green]+ Entities added:[/green]")
                for e in entities["added"]:
                    console.print(f"    {e['name']} ({e['type']})")

            if entities["removed"]:
                console.print("[red]- Entities removed:[/red]")
                for e in entities["removed"]:
                    console.print(f"    {e['name']} ({e['type']})")

            if entities["modified"]:
                console.print("[yellow]~ Entities modified:[/yellow]")
                for e in entities["modified"]:
                    changes = e["changes"]
                    change_str = ", ".join(f"{k}" for k in changes.keys())
                    console.print(f"    {e['name']}: {change_str}")

            if relations["added"]:
                console.print("[green]+ Relations added:[/green]")
                for r in relations["added"]:
                    console.print(f"    {r['from_entity'][:8]}... → {r['type']} → {r['to_entity'][:8]}...")

            if relations["removed"]:
                console.print("[red]- Relations removed:[/red]")
                for r in relations["removed"]:
                    console.print(f"    {r['from_entity'][:8]}... → {r['type']} → {r['to_entity'][:8]}...")

            if not any([entities["added"], entities["removed"], entities["modified"],
                       relations["added"], relations["removed"]]):
                console.print("[dim]No changes in this time range[/dim]")

    except ValueError as e:
        console.print(f"[red]Error:[/red] {e}")


@cli.command()
@click.argument("entity_name")
@click.option("--json", "as_json", is_flag=True, help="Output as JSON")
@click.pass_context
def history(ctx, entity_name, as_json):
    """Show the history of changes to an entity."""
    memory_dir = ctx.obj["memory_dir"]

    try:
        engine = get_engine(memory_dir)
        events = engine.get_entity_history(entity_name)

        if not events:
            console.print(f"[yellow]No history found for entity '{entity_name}'[/yellow]")
            return

        if as_json:
            console.print(json.dumps(events, indent=2))
        else:
            console.print(f"[bold]History for '{entity_name}'[/bold]")
            console.print()

            for event in events:
                ts = datetime.fromisoformat(event["timestamp"])
                rel_time = format_relative_time(ts)
                op = event["operation"]

                if op == "create_entity":
                    console.print(f"[green]+[/green] {ts.strftime('%Y-%m-%d %H:%M')} ({rel_time})")
                    console.print(f"    Created as {event['data'].get('type', 'entity')}")
                elif op == "add_observation":
                    obs_text = event["data"].get("observation", {}).get("text", "")[:50]
                    console.print(f"[blue]~[/blue] {ts.strftime('%Y-%m-%d %H:%M')} ({rel_time})")
                    console.print(f"    Added observation: {obs_text}...")
                elif op == "delete_observation":
                    console.print(f"[red]-[/red] {ts.strftime('%Y-%m-%d %H:%M')} ({rel_time})")
                    console.print(f"    Removed observation")
                else:
                    console.print(f"[yellow]~[/yellow] {ts.strftime('%Y-%m-%d %H:%M')} ({rel_time})")
                    console.print(f"    {op}")

    except ValueError as e:
        console.print(f"[red]Error:[/red] {e}")


# ─────────────────────────────────────────────────────────────────────────────
# Recovery Commands
# ─────────────────────────────────────────────────────────────────────────────


@cli.command()
@click.option("--json", "as_json", is_flag=True, help="Output as JSON")
@click.pass_context
def reload(ctx, as_json):
    """Reload graph state from mnemograph.db on disk."""
    memory_dir = ctx.obj["memory_dir"]
    engine = get_engine(memory_dir)

    result = engine.reload()

    if as_json:
        console.print(json.dumps(result, indent=2, default=str))
    else:
        console.print(f"[green]✓[/green] Reloaded: {result['entities']} entities, {result['relations']} relations")
        console.print(f"   Processed {result['events_processed']} events")


@cli.command()
@click.option("--steps", "-n", default=1, type=int, help="Go back N commits (default: 1)")
@click.option("--to-commit", "-c", help="Restore to specific commit hash")
@click.option("--yes", "-y", is_flag=True, help="Skip confirmation prompt")
@click.option("--json", "as_json", is_flag=True, help="Output as JSON")
@click.pass_context
def rewind(ctx, steps, to_commit, yes, as_json):
    """Rewind graph to a previous state using git."""
    memory_dir = ctx.obj["memory_dir"]
    engine = get_engine(memory_dir)

    if not engine._in_git_repo:
        console.print("[red]Error:[/red] Not in a git repository. Cannot use git-based rewind.")
        console.print("[dim]Tip: Use 'mnemograph restore --to <timestamp>' for event-based restore.[/dim]")
        return

    if not yes and not as_json:
        if to_commit:
            console.print(f"[yellow]⚠  This will rewind mnemograph.db to commit {to_commit}.[/yellow]")
        else:
            console.print(f"[yellow]⚠  This will rewind mnemograph.db by {steps} commit(s).[/yellow]")
        console.print("[dim]   Audit trail will only be in git (not in events).[/dim]")
        console.print()
        if not click.confirm("Continue?", default=False):
            console.print("[dim]Cancelled.[/dim]")
            return

    result = engine.rewind(steps=steps, to_commit=to_commit)

    if as_json:
        console.print(json.dumps(result, indent=2, default=str))
    elif result.get("status") == "error":
        console.print(f"[red]Error:[/red] {result['error']}")
        if "tip" in result:
            console.print(f"[dim]Tip: {result['tip']}[/dim]")
    else:
        console.print(f"[green]✓[/green] Rewound to commit {result['restored_to_commit']}")
        console.print(f"   Now at: {result['entities']} entities, {result['relations']} relations")
        console.print("[dim]   Tip: Run 'git diff' to see what changed[/dim]")


@cli.command()
@click.option("--to", "timestamp", required=True, help="Timestamp to restore to")
@click.option("--reason", "-m", help="Reason for restoring")
@click.option("--yes", "-y", is_flag=True, help="Skip confirmation prompt")
@click.option("--json", "as_json", is_flag=True, help="Output as JSON")
@click.pass_context
def restore(ctx, timestamp, reason, yes, as_json):
    """Restore graph to state at a specific timestamp."""
    memory_dir = ctx.obj["memory_dir"]
    engine = get_engine(memory_dir)

    try:
        ts = parse_time_reference(timestamp)
        preview_state = engine.state_at(ts)
        preview_entities = len(preview_state.entities)
        preview_relations = len(preview_state.relations)
    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        return

    if preview_entities == 0:
        console.print(f"[red]Error:[/red] No entities found at {timestamp}")
        console.print("[dim]Tip: Use 'mnemograph show --at <timestamp>' to explore available states.[/dim]")
        return

    if not yes and not as_json:
        current_entities = len(engine.state.entities)
        current_relations = len(engine.state.relations)
        console.print(f"[yellow]⚠  Restore to {timestamp}[/yellow]")
        console.print(f"   Current: {current_entities} entities, {current_relations} relations")
        console.print(f"   After:   {preview_entities} entities, {preview_relations} relations")
        console.print("[dim]   Events will record: clear + recreate (full audit trail)[/dim]")
        console.print()
        if not click.confirm("Continue?", default=False):
            console.print("[dim]Cancelled.[/dim]")
            return

    result = engine.restore_state_at(timestamp=timestamp, reason=reason or "")

    if as_json:
        console.print(json.dumps(result, indent=2, default=str))
    elif result.get("status") == "error":
        console.print(f"[red]Error:[/red] {result['error']}")
        if "tip" in result:
            console.print(f"[dim]Tip: {result['tip']}[/dim]")
    else:
        console.print(f"[green]✓[/green] Restored to {result['restored_to']}")
        console.print(f"   Now at: {result['entities']} entities, {result['relations']} relations")
        if reason:
            console.print(f"   Reason: {reason}")
        console.print("[dim]   Full audit trail preserved in events[/dim]")


# ─────────────────────────────────────────────────────────────────────────────
# Graph Health Commands
# ─────────────────────────────────────────────────────────────────────────────


@cli.command()
@click.option("--fix", is_flag=True, help="Interactive cleanup mode")
@click.option("--json", "as_json", is_flag=True, help="Output as JSON")
@click.pass_context
def health(ctx, fix, as_json):
    """Show graph health report."""
    memory_dir = ctx.obj["memory_dir"]
    engine = get_engine(memory_dir)

    report = engine.get_graph_health()

    if as_json:
        console.print(json.dumps(report, indent=2, default=str))
        return

    summary = report["summary"]

    console.print("[bold]Graph Health Report[/bold]")
    console.print("═" * 40)
    console.print(f"Entities: {summary['total_entities']}    Relations: {summary['total_relations']}")
    console.print()

    issues_found = (
        summary["orphan_count"]
        + summary["duplicate_groups"]
        + summary["overloaded_count"]
        + summary["weak_relation_count"]
    )

    if issues_found:
        console.print("[yellow]Issues Found:[/yellow]")
        if summary["orphan_count"]:
            console.print(f"  ⚠ {summary['orphan_count']} orphan entities (no relations)")
        if summary["duplicate_groups"]:
            console.print(f"  ⚠ {summary['duplicate_groups']} potential duplicate groups")
        if summary["overloaded_count"]:
            console.print(f"  ⚠ {summary['overloaded_count']} overloaded entities (>15 observations)")
        if summary["weak_relation_count"]:
            console.print(f"  ⚠ {summary['weak_relation_count']} weak relations (may be noise)")
        if summary["cluster_count"] > 1:
            console.print(f"  ⚠ {summary['cluster_count']} disconnected clusters")
        console.print()
    else:
        console.print("[green]✓ No issues found[/green]")
        console.print()

    console.print("[bold]Recommendations:[/bold]")
    for rec in report["recommendations"]:
        console.print(f"  • {rec}")

    if fix and issues_found:
        console.print()
        console.print("[cyan]Interactive cleanup mode...[/cyan]")
        _interactive_cleanup(engine, report)
    elif fix and not issues_found:
        console.print()
        console.print("[green]Nothing to fix![/green]")


def _interactive_cleanup(engine: MemoryEngine, report: dict):
    """Interactive mode for fixing graph issues."""
    issues = report["issues"]

    if issues["orphans"]:
        console.print()
        console.print(f"[bold]Found {len(issues['orphans'])} orphan entities:[/bold]")

        for orphan in issues["orphans"][:10]:
            console.print()
            console.print(f"  [cyan]{orphan['name']}[/cyan] ({orphan['type']})")
            console.print(f"  {orphan['observation_count']} observations, created {orphan['created_at'][:10]}")

            choice = click.prompt(
                "  [c]onnect  [d]elete  [s]kip",
                type=click.Choice(["c", "d", "s"], case_sensitive=False),
                default="s",
            )

            if choice == "c":
                suggestions = engine.suggest_relations(orphan["name"], limit=3)
                if suggestions and "error" not in suggestions[0]:
                    console.print("  Suggested connections:")
                    for i, s in enumerate(suggestions, 1):
                        console.print(f"    {i}. {s['target']} ({s['suggested_relation']}, {s['confidence']:.0%})")

                    target = click.prompt("  Connect to", default=suggestions[0]["target"] if suggestions else "")
                    rel_type = click.prompt("  Relation type", default=suggestions[0]["suggested_relation"] if suggestions else "related_to")

                    if target:
                        engine.create_relations([{
                            "from": orphan["name"],
                            "to": target,
                            "relationType": rel_type,
                        }])
                        console.print(f"  [green]✓[/green] Created relation: {orphan['name']} → {target}")
                else:
                    target = click.prompt("  Connect to (entity name)")
                    rel_type = click.prompt("  Relation type", default="related_to")
                    if target:
                        engine.create_relations([{
                            "from": orphan["name"],
                            "to": target,
                            "relationType": rel_type,
                        }])
                        console.print(f"  [green]✓[/green] Created relation")

            elif choice == "d":
                if click.confirm(f"  Delete '{orphan['name']}'?", default=False):
                    engine.delete_entities([orphan["name"]])
                    console.print(f"  [green]✓[/green] Deleted {orphan['name']}")

    if issues["potential_duplicates"]:
        console.print()
        console.print(f"[bold]Found {len(issues['potential_duplicates'])} potential duplicate groups:[/bold]")

        for group in issues["potential_duplicates"][:5]:
            console.print()
            console.print(f"  [cyan]{group['entity']}[/cyan] similar to: {', '.join(group['similar_to'])}")

            choice = click.prompt(
                "  [m]erge  [s]kip",
                type=click.Choice(["m", "s"], case_sensitive=False),
                default="s",
            )

            if choice == "m":
                target = click.prompt("  Merge into", default=group["entity"])
                source = click.prompt("  Merge from", default=group["similar_to"][0] if group["similar_to"] else "")

                if source and target and source != target:
                    result = engine.merge_entities(source, target)
                    if "error" in result:
                        console.print(f"  [red]✗[/red] {result['error']}")
                    else:
                        console.print(f"  [green]✓[/green] Merged {source} → {target}")

    console.print()
    console.print("[green]Cleanup complete.[/green]")


@cli.command()
@click.argument("name")
@click.option("-t", "--threshold", type=float, default=0.7, help="Similarity threshold 0-1")
@click.option("--json", "as_json", is_flag=True, help="Output as JSON")
@click.pass_context
def similar(ctx, name, threshold, as_json):
    """Find entities similar to a given name."""
    memory_dir = ctx.obj["memory_dir"]
    engine = get_engine(memory_dir)

    results = engine.find_similar(name, threshold=threshold)

    if as_json:
        console.print(json.dumps(results, indent=2, default=str))
        return

    if not results:
        console.print(f"[green]✓[/green] No similar entities found for '{name}'")
        console.print("[dim]   Safe to create a new entity with this name.[/dim]")
        return

    console.print(f"[bold]Entities similar to '{name}':[/bold]")
    console.print()

    table = Table()
    table.add_column("Name", style="cyan")
    table.add_column("Type")
    table.add_column("Similarity", justify="right", style="green")
    table.add_column("Observations", justify="right")

    for r in results:
        sim_display = f"{r['similarity']:.0%}"
        if r["similarity"] >= 0.85:
            sim_display = f"[red]{sim_display}[/red] ← consider merging"
        table.add_row(r["name"], r["type"], sim_display, str(r["observation_count"]))

    console.print(table)


@cli.command()
@click.option("--json", "as_json", is_flag=True, help="Output as JSON")
@click.pass_context
def orphans(ctx, as_json):
    """Find entities with no relations."""
    memory_dir = ctx.obj["memory_dir"]
    engine = get_engine(memory_dir)

    results = engine.find_orphans()

    if as_json:
        console.print(json.dumps(results, indent=2, default=str))
        return

    if not results:
        console.print("[green]✓[/green] No orphan entities found")
        console.print("[dim]   All entities are connected.[/dim]")
        return

    console.print(f"[bold]Orphan entities ({len(results)} found):[/bold]")
    console.print()

    table = Table()
    table.add_column("Name", style="cyan")
    table.add_column("Type")
    table.add_column("Observations", justify="right")
    table.add_column("Created", style="dim")

    for r in results:
        table.add_row(r["name"], r["type"], str(r["observation_count"]), r["created_at"][:10])

    console.print(table)
    console.print()
    console.print("[dim]Run 'mnemograph health --fix' for interactive cleanup.[/dim]")


@cli.command()
@click.argument("entity")
@click.option("-n", "--limit", type=int, default=5, help="Max suggestions")
@click.option("--json", "as_json", is_flag=True, help="Output as JSON")
@click.pass_context
def suggest(ctx, entity, limit, as_json):
    """Suggest potential relations for an entity."""
    memory_dir = ctx.obj["memory_dir"]
    engine = get_engine(memory_dir)

    results = engine.suggest_relations(entity, limit=limit)

    if as_json:
        console.print(json.dumps(results, indent=2, default=str))
        return

    if results and "error" in results[0]:
        console.print(f"[red]✗[/red] {results[0]['error']}")
        return

    if not results:
        console.print(f"[yellow]![/yellow] No relation suggestions for '{entity}'")
        return

    console.print(f"[bold]Suggested relations for '{entity}':[/bold]")
    console.print()

    table = Table()
    table.add_column("Target", style="cyan")
    table.add_column("Relation Type")
    table.add_column("Confidence", justify="right", style="green")
    table.add_column("Reason", style="dim")

    for r in results:
        table.add_row(r["target"], r["suggested_relation"], f"{r['confidence']:.0%}", r.get("reason", ""))

    console.print(table)


# ─────────────────────────────────────────────────────────────────────────────
# Relation Weight Commands
# ─────────────────────────────────────────────────────────────────────────────


@cli.group()
def relation():
    """Commands for relation weights and connections."""
    pass


@relation.command("show")
@click.argument("relation_id")
@click.option("--json", "as_json", is_flag=True, help="Output as JSON")
@click.pass_context
def relation_show(ctx, relation_id, as_json):
    """Show relation details with weight breakdown."""
    memory_dir = ctx.obj["memory_dir"]
    engine = get_engine(memory_dir)

    result = engine.get_relation_weight(relation_id)
    if not result:
        console.print(f"[red]Relation not found:[/red] {relation_id}")
        return

    if as_json:
        console.print(json.dumps(result, indent=2))
    else:
        console.print(f"[bold]Relation:[/bold] {result['relation_id'][:12]}...")
        console.print(f"  From: [cyan]{result['from_entity']}[/cyan]")
        console.print(f"  To:   [cyan]{result['to_entity']}[/cyan]")
        console.print(f"  Type: {result['relation_type']}")
        console.print()

        console.print(f"[bold]Combined Weight:[/bold] [green]{result['combined_weight']:.3f}[/green]")
        console.print()

        comp = result["components"]
        console.print("[bold]Weight Components:[/bold]")
        console.print(f"  Recency:   {comp['recency']:.3f} [dim](40%)[/dim]")
        console.print(f"  Co-access: {comp['co_access']:.3f} [dim](30%)[/dim]")
        console.print(f"  Explicit:  {comp['explicit']:.3f} [dim](30%)[/dim]")
        console.print()

        meta = result["metadata"]
        console.print("[bold]Metadata:[/bold]")
        console.print(f"  Access count: {meta['access_count']}")
        console.print(f"  Last accessed: {meta['last_accessed']}")
        console.print(f"  Created at: {meta['created_at']}")


@relation.command("set-weight")
@click.argument("relation_id")
@click.argument("weight", type=float)
@click.pass_context
def relation_set_weight(ctx, relation_id, weight):
    """Set explicit importance weight for a relation."""
    memory_dir = ctx.obj["memory_dir"]
    engine = get_engine(memory_dir)

    try:
        result = engine.set_relation_importance(relation_id, weight)
        if not result:
            console.print(f"[red]Relation not found:[/red] {relation_id}")
            return

        console.print(f"[green]✓[/green] Set explicit weight to {weight:.2f}")
        console.print(f"  New combined weight: [green]{result['combined_weight']:.3f}[/green]")
    except ValueError as e:
        console.print(f"[red]Error:[/red] {e}")


@relation.command("list")
@click.option("--max-weight", type=float, default=0.1, help="Only show relations with weight <= this value")
@click.option("-n", "--limit", type=int, default=20, help="Max relations to show")
@click.option("--json", "as_json", is_flag=True, help="Output as JSON")
@click.pass_context
def relation_list(ctx, max_weight, limit, as_json):
    """List weak relations (candidates for pruning)."""
    memory_dir = ctx.obj["memory_dir"]
    engine = get_engine(memory_dir)

    results = engine.get_weak_relations(max_weight=max_weight, limit=limit)

    if as_json:
        console.print(json.dumps(results, indent=2))
    else:
        if not results:
            console.print(f"[green]No relations with weight ≤ {max_weight}[/green]")
            return

        console.print(f"[bold]Weak Relations (weight ≤ {max_weight}):[/bold]")
        console.print()

        table = Table()
        table.add_column("From", style="cyan")
        table.add_column("Type")
        table.add_column("To", style="cyan")
        table.add_column("Weight", justify="right", style="yellow")

        for r in results:
            table.add_row(r["from_entity"][:20], r["relation_type"], r["to_entity"][:20], f"{r['weight']:.3f}")

        console.print(table)
        console.print()
        console.print(f"[dim]Found {len(results)} weak relation(s)[/dim]")


@cli.command()
@click.argument("entity_name")
@click.option("-n", "--limit", type=int, default=10, help="Max connections to show")
@click.option("--json", "as_json", is_flag=True, help="Output as JSON")
@click.pass_context
def connections(ctx, entity_name, limit, as_json):
    """Show an entity's strongest connections by weight."""
    memory_dir = ctx.obj["memory_dir"]
    engine = get_engine(memory_dir)

    results = engine.get_strongest_connections(entity_name, limit=limit)

    if not results:
        console.print(f"[yellow]No connections found for '{entity_name}'[/yellow]")
        return

    if as_json:
        console.print(json.dumps(results, indent=2))
    else:
        console.print(f"[bold]Strongest connections for '{entity_name}':[/bold]")
        console.print()

        table = Table()
        table.add_column("Connected To", style="cyan")
        table.add_column("Relation Type")
        table.add_column("Weight", justify="right", style="green")
        table.add_column("Recency", justify="right", style="dim")
        table.add_column("Co-access", justify="right", style="dim")
        table.add_column("Explicit", justify="right", style="dim")

        for conn in results:
            comp = conn["components"]
            table.add_row(
                conn["connected_to"][:25],
                conn["relation_type"],
                f"{conn['weight']:.3f}",
                f"{comp['recency']:.2f}",
                f"{comp['co_access']:.2f}",
                f"{comp['explicit']:.2f}",
            )

        console.print(table)


# ─────────────────────────────────────────────────────────────────────────────
# Destructive Commands
# ─────────────────────────────────────────────────────────────────────────────


@cli.command()
@click.option("--reason", "-m", help="Reason for clearing")
@click.option("--yes", "-y", is_flag=True, help="Skip confirmation prompt")
@click.option("--json", "as_json", is_flag=True, help="Output as JSON")
@click.pass_context
def clear(ctx, reason, yes, as_json):
    """Clear all entities and relations from the graph."""
    memory_dir = ctx.obj["memory_dir"]
    engine = get_engine(memory_dir)

    entity_count = len(engine.state.entities)
    relation_count = len(engine.state.relations)

    if entity_count == 0 and relation_count == 0:
        if as_json:
            console.print(json.dumps({"status": "empty", "message": "Graph is already empty"}, indent=2))
        else:
            console.print("[yellow]![/yellow] Graph is already empty. Nothing to clear.")
        return

    if not yes:
        console.print(f"[yellow]⚠  This will clear ALL {entity_count} entities and {relation_count} relations.[/yellow]")
        console.print("[dim]   (Event is recorded — can rewind with 'mnemograph show --at <timestamp>')[/dim]")
        console.print()
        if not click.confirm("Are you sure you want to clear the graph?", default=False):
            console.print("[dim]Cancelled.[/dim]")
            return

    result = engine.clear_graph(reason=reason or "")

    if as_json:
        console.print(json.dumps(result, indent=2, default=str))
    else:
        console.print(f"[green]✓[/green] Cleared {result['entities_cleared']} entities, {result['relations_cleared']} relations")
        if reason:
            console.print(f"   Reason: {reason}")
        console.print("[dim]   Tip: Use 'mnemograph show --at <timestamp>' to view graph before clear[/dim]")


@cli.command()
@click.option("--delete-entity", "-e", multiple=True, help="Entity name(s) to delete completely")
@click.option("--reason", "-m", help="Reason for compacting")
@click.option("--yes", "-y", is_flag=True, help="Skip confirmation prompt")
@click.option("--json", "as_json", is_flag=True, help="Output as JSON")
@click.pass_context
def compact(ctx, delete_entity, reason, yes, as_json):
    """Compact graph: materialize current state, apply deletions, regenerate events."""
    import subprocess
    memory_dir = ctx.obj["memory_dir"]
    engine = get_engine(memory_dir)

    current_entities = len(engine.state.entities)
    current_relations = len(engine.state.relations)
    current_events = len(engine.event_store.read_all())

    if current_entities == 0 and current_relations == 0:
        if as_json:
            console.print(json.dumps({"status": "empty", "message": "Graph is already empty"}, indent=2))
        else:
            console.print("[yellow]![/yellow] Graph is already empty. Nothing to compact.")
        return

    deleted_info = []
    if delete_entity:
        for name in delete_entity:
            entity_id = engine._resolve_entity(name)
            if entity_id:
                entity = engine.state.entities[entity_id]
                rel_count = sum(
                    1 for r in engine.state.relations
                    if r.from_entity == entity_id or r.to_entity == entity_id
                )
                deleted_info.append(f"  - {entity.name} ({entity.type}) + {rel_count} relations")
            else:
                deleted_info.append(f"  - {name} [dim](not found)[/dim]")

    if not yes and not as_json:
        console.print("[yellow]⚠  This will REWRITE the entire event history.[/yellow]")
        console.print(f"   Current: {current_entities} entities, {current_relations} relations, {current_events} events")

        if deleted_info:
            console.print("\n   Will delete:")
            for info in deleted_info:
                console.print(info)

        if engine._in_git_repo:
            console.print("\n[dim]   Pre-compaction state will be auto-committed to git.[/dim]")
        else:
            console.print("\n[yellow]   Warning: Not in git repo — no safety backup![/yellow]")

        console.print()
        if not click.confirm("Continue?", default=False):
            console.print("[dim]Cancelled.[/dim]")
            return

    # Auto-commit to git if available
    if engine._in_git_repo and engine._git_root:
        events_path = engine.event_store.db_path
        relative_path = events_path.relative_to(engine._git_root)

        subprocess.run(["git", "add", str(relative_path)], cwd=engine._git_root, capture_output=True)
        commit_msg = f"Pre-compaction snapshot"
        if reason:
            commit_msg += f": {reason}"
        subprocess.run(
            ["git", "commit", "-m", commit_msg, "--allow-empty"],
            cwd=engine._git_root,
            capture_output=True,
            text=True,
        )

    # Get current state
    state = engine.state

    # Apply deletions
    deleted_entities = []
    deleted_relation_count = 0

    for name in delete_entity:
        entity_id = engine._resolve_entity(name)
        if entity_id and entity_id in state.entities:
            deleted_entities.append(state.entities[entity_id].name)
            del state.entities[entity_id]
            if name in state._name_to_id:
                del state._name_to_id[name]

            for rel in list(state.relations):
                if rel.from_entity == entity_id or rel.to_entity == entity_id:
                    state.relations.remove(rel)
                    deleted_relation_count += 1

    # Generate minimal event sequence
    new_events = []

    compact_event = MemoryEvent(
        op="compact",
        session_id="cli",
        source="user",
        data={
            "reason": reason or "",
            "deleted_entities": deleted_entities,
            "deleted_relation_count": deleted_relation_count,
            "original_event_count": current_events,
        },
    )
    new_events.append(compact_event)

    for entity in state.entities.values():
        event = MemoryEvent(
            op="create_entity",
            session_id="compact",
            source="user",
            data=entity.model_dump(mode="json"),
        )
        event.ts = entity.created_at
        new_events.append(event)

    for relation_obj in state.relations:
        event = MemoryEvent(
            op="create_relation",
            session_id="compact",
            source="user",
            data=relation_obj.model_dump(mode="json"),
        )
        event.ts = relation_obj.created_at
        new_events.append(event)

    # Write new events in transaction
    conn = engine.event_store.get_connection()
    try:
        conn.execute("BEGIN IMMEDIATE")
        engine.event_store.clear(commit=False)
        engine.event_store.append_batch(new_events, commit=False)
        conn.commit()
    except Exception as e:
        conn.rollback()
        raise click.ClickException(f"Compaction failed, rolled back: {e}")

    engine.reload()

    result = {
        "status": "compacted",
        "deleted_entities": deleted_entities,
        "deleted_relation_count": deleted_relation_count,
        "entities_before": current_entities,
        "entities_after": len(engine.state.entities),
        "relations_before": current_relations,
        "relations_after": len(engine.state.relations),
        "events_before": current_events,
        "events_after": len(new_events),
    }

    if as_json:
        console.print(json.dumps(result, indent=2, default=str))
    else:
        console.print(f"[green]✓[/green] Compacted: {result['events_before']} → {result['events_after']} events")
        console.print(f"   Entities: {result['entities_after']}, Relations: {result['relations_after']}")
        if deleted_entities:
            console.print(f"   Deleted: {', '.join(deleted_entities)}")
        if engine._in_git_repo:
            console.print("[dim]   Pre-compaction state saved in git[/dim]")


# ─────────────────────────────────────────────────────────────────────────────
# Branch Commands
# ─────────────────────────────────────────────────────────────────────────────


@cli.group()
def branch():
    """Branch management commands."""
    pass


@branch.command("list")
@click.option("--all", "-a", is_flag=True, help="Include archived branches")
@click.option("--json", "as_json", is_flag=True, help="Output as JSON")
@click.pass_context
def branch_list(ctx, all, as_json):
    """List all branches."""
    memory_dir = ctx.obj["memory_dir"]

    if not memory_dir.exists():
        console.print(f"[red]Memory directory does not exist:[/red] {memory_dir}")
        return

    engine = get_engine(memory_dir)
    branches = engine.branch_manager.list(include_archived=all)
    current = engine.branch_manager.current_branch_name()

    if as_json:
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
        console.print(json.dumps(result, indent=2))
    else:
        for b in branches:
            marker = "*" if b.name == current else " "
            if b.name == "main":
                entity_count = len(engine.state.entities)
                rel_count = len(engine.state.relations)
            else:
                entity_count = len(b.entity_ids)
                rel_count = len(b.relation_ids)
            status = "" if b.is_active else " [archived]"
            console.print(f"{marker} {b.name:<30} ({entity_count} entities, {rel_count} relations){status}")


@branch.command("show")
@click.argument("name", required=False)
@click.option("--verbose", "-v", is_flag=True, help="Show entity details")
@click.option("--json", "as_json", is_flag=True, help="Output as JSON")
@click.pass_context
def branch_show(ctx, name, verbose, as_json):
    """Show details of a branch."""
    memory_dir = ctx.obj["memory_dir"]

    if not memory_dir.exists():
        console.print(f"[red]Memory directory does not exist:[/red] {memory_dir}")
        return

    engine = get_engine(memory_dir)
    branch_name = name or engine.branch_manager.current_branch_name()

    try:
        branch_obj = engine.branch_manager.get(branch_name)
    except ValueError as e:
        console.print(f"[red]Error:[/red] {e}")
        return

    if as_json:
        result = branch_obj.to_dict()
        result["is_current"] = branch_name == engine.branch_manager.current_branch_name()
        console.print(json.dumps(result, indent=2, default=str))
    else:
        current = engine.branch_manager.current_branch_name()
        marker = " (current)" if branch_name == current else ""
        console.print(f"Branch: {branch_obj.name}{marker}")
        console.print(f"Description: {branch_obj.description or '(none)'}")
        console.print(f"Parent: {branch_obj.parent or '(none)'}")
        console.print(f"Created: {branch_obj.created_at.strftime('%Y-%m-%d %H:%M:%S')}")
        console.print(f"Active: {branch_obj.is_active}")

        if branch_obj.name == "main":
            console.print(f"Entities: {len(engine.state.entities)} (all)")
            console.print(f"Relations: {len(engine.state.relations)} (all)")
        else:
            console.print(f"Entities: {len(branch_obj.entity_ids)}")
            console.print(f"Relations: {len(branch_obj.relation_ids)}")

            if verbose and branch_obj.entity_ids:
                console.print("\nEntity IDs:")
                for eid in sorted(branch_obj.entity_ids)[:20]:
                    entity = engine.state.entities.get(eid)
                    entity_name = entity.name if entity else "(deleted)"
                    console.print(f"  {eid[:8]}  {entity_name}")
                if len(branch_obj.entity_ids) > 20:
                    console.print(f"  ... and {len(branch_obj.entity_ids) - 20} more")


@branch.command("create")
@click.argument("name")
@click.option("--seeds", "-s", multiple=True, help="Seed entity names")
@click.option("--depth", "-d", type=int, default=2, help="Expansion depth")
@click.option("--description", "-m", help="Branch description")
@click.option("--checkout", "-c", is_flag=True, help="Switch to new branch")
@click.pass_context
def branch_create(ctx, name, seeds, depth, description, checkout):
    """Create a new branch."""
    memory_dir = ctx.obj["memory_dir"]

    if not memory_dir.exists():
        console.print(f"[red]Memory directory does not exist:[/red] {memory_dir}")
        return

    engine = get_engine(memory_dir)
    seed_list = list(seeds) if seeds else []

    try:
        branch_obj = engine.branch_manager.create(
            name=name,
            seed_entities=seed_list,
            description=description or "",
            depth=depth,
        )
    except ValueError as e:
        console.print(f"[red]Error:[/red] {e}")
        return

    if checkout:
        engine.branch_manager.checkout(name)
        console.print(f"[green]✓[/green] Created and switched to branch '{name}'")
    else:
        console.print(f"[green]✓[/green] Created branch '{name}'")

    console.print(f"  Entities: {len(branch_obj.entity_ids)}")
    console.print(f"  Relations: {len(branch_obj.relation_ids)}")


@branch.command("delete")
@click.argument("name")
@click.option("--yes", "-y", is_flag=True, help="Skip confirmation")
@click.pass_context
def branch_delete(ctx, name, yes):
    """Delete a branch."""
    memory_dir = ctx.obj["memory_dir"]

    if not memory_dir.exists():
        console.print(f"[red]Memory directory does not exist:[/red] {memory_dir}")
        return

    engine = get_engine(memory_dir)

    if not yes:
        if not click.confirm(f"Delete branch '{name}'?", default=False):
            console.print("[dim]Cancelled.[/dim]")
            return

    try:
        engine.branch_manager.delete(name)
        console.print(f"[green]✓[/green] Deleted branch '{name}'")
    except ValueError as e:
        console.print(f"[red]Error:[/red] {e}")


@branch.command("checkout")
@click.argument("name")
@click.pass_context
def branch_checkout(ctx, name):
    """Switch to a branch."""
    memory_dir = ctx.obj["memory_dir"]

    if not memory_dir.exists():
        console.print(f"[red]Memory directory does not exist:[/red] {memory_dir}")
        return

    engine = get_engine(memory_dir)

    try:
        branch_obj = engine.branch_manager.checkout(name)
        console.print(f"[green]✓[/green] Switched to branch '{name}'")
        if branch_obj.name != "main":
            console.print(f"  Entities: {len(branch_obj.entity_ids)}")
            console.print(f"  Relations: {len(branch_obj.relation_ids)}")
    except ValueError as e:
        console.print(f"[red]Error:[/red] {e}")


# Top-level checkout shortcut
@cli.command("checkout")
@click.argument("name")
@click.pass_context
def checkout(ctx, name):
    """Switch to a branch (shortcut for 'branch checkout')."""
    ctx.invoke(branch_checkout, name=name)


@branch.command("add")
@click.argument("entities", nargs=-1, required=True)
@click.option("--no-relations", is_flag=True, help="Don't include relations")
@click.pass_context
def branch_add(ctx, entities, no_relations):
    """Add entities to the current branch."""
    memory_dir = ctx.obj["memory_dir"]

    if not memory_dir.exists():
        console.print(f"[red]Memory directory does not exist:[/red] {memory_dir}")
        return

    engine = get_engine(memory_dir)
    current = engine.branch_manager.current_branch_name()

    if current == "main":
        console.print("[red]Cannot add entities to main branch (main sees everything).[/red]")
        return

    entity_ids = []
    for name in entities:
        eid = engine._resolve_entity(name)
        if eid:
            entity_ids.append(eid)
        else:
            console.print(f"[yellow]Warning:[/yellow] Entity not found: {name}")

    if not entity_ids:
        console.print("[red]No valid entities to add.[/red]")
        return

    try:
        branch_obj = engine.branch_manager.add_entities(
            current,
            entity_ids,
            include_relations=not no_relations,
        )
        console.print(f"[green]✓[/green] Added {len(entity_ids)} entities to '{current}'")
        console.print(f"  Total entities: {len(branch_obj.entity_ids)}")
        console.print(f"  Total relations: {len(branch_obj.relation_ids)}")
    except ValueError as e:
        console.print(f"[red]Error:[/red] {e}")


@branch.command("remove")
@click.argument("entities", nargs=-1, required=True)
@click.option("--keep-relations", is_flag=True, help="Don't cascade relation removal")
@click.pass_context
def branch_remove(ctx, entities, keep_relations):
    """Remove entities from the current branch."""
    memory_dir = ctx.obj["memory_dir"]

    if not memory_dir.exists():
        console.print(f"[red]Memory directory does not exist:[/red] {memory_dir}")
        return

    engine = get_engine(memory_dir)
    current = engine.branch_manager.current_branch_name()

    if current == "main":
        console.print("[red]Cannot remove entities from main branch.[/red]")
        return

    entity_ids = []
    for name in entities:
        eid = engine._resolve_entity(name)
        if eid:
            entity_ids.append(eid)
        else:
            console.print(f"[yellow]Warning:[/yellow] Entity not found: {name}")

    if not entity_ids:
        console.print("[red]No valid entities to remove.[/red]")
        return

    try:
        branch_obj = engine.branch_manager.remove_entities(
            current,
            entity_ids,
            cascade_relations=not keep_relations,
        )
        console.print(f"[green]✓[/green] Removed {len(entity_ids)} entities from '{current}'")
        console.print(f"  Remaining entities: {len(branch_obj.entity_ids)}")
        console.print(f"  Remaining relations: {len(branch_obj.relation_ids)}")
    except ValueError as e:
        console.print(f"[red]Error:[/red] {e}")


@branch.command("diff")
@click.argument("branch_a", required=False)
@click.argument("branch_b", required=True)
@click.option("--json", "as_json", is_flag=True, help="Output as JSON")
@click.pass_context
def branch_diff(ctx, branch_a, branch_b, as_json):
    """Show diff between two branches."""
    memory_dir = ctx.obj["memory_dir"]

    if not memory_dir.exists():
        console.print(f"[red]Memory directory does not exist:[/red] {memory_dir}")
        return

    engine = get_engine(memory_dir)
    a = branch_a or engine.branch_manager.current_branch_name()

    try:
        diff_result = engine.branch_manager.diff(a, branch_b)
    except ValueError as e:
        console.print(f"[red]Error:[/red] {e}")
        return

    if as_json:
        console.print(json.dumps(diff_result, indent=2))
    else:
        console.print(f"Diff: {diff_result['branch_a']} vs {diff_result['branch_b']}")
        console.print()

        if diff_result["only_in_a"]["entities"]:
            console.print(f"Only in {diff_result['branch_a']} ({len(diff_result['only_in_a']['entities'])} entities):")
            for eid in list(diff_result["only_in_a"]["entities"])[:10]:
                entity = engine.state.entities.get(eid)
                entity_name = entity.name if entity else eid[:8]
                console.print(f"  - {entity_name}")
            if len(diff_result["only_in_a"]["entities"]) > 10:
                console.print(f"  ... and {len(diff_result['only_in_a']['entities']) - 10} more")
            console.print()

        if diff_result["only_in_b"]["entities"]:
            console.print(f"Only in {diff_result['branch_b']} ({len(diff_result['only_in_b']['entities'])} entities):")
            for eid in list(diff_result["only_in_b"]["entities"])[:10]:
                entity = engine.state.entities.get(eid)
                entity_name = entity.name if entity else eid[:8]
                console.print(f"  + {entity_name}")
            if len(diff_result["only_in_b"]["entities"]) > 10:
                console.print(f"  ... and {len(diff_result['only_in_b']['entities']) - 10} more")
            console.print()

        if diff_result["in_both"]["entities"]:
            console.print(f"In both ({len(diff_result['in_both']['entities'])} entities)")


@branch.command("merge")
@click.argument("source")
@click.option("--target", "-t", help="Target branch (default: current)")
@click.pass_context
def branch_merge(ctx, source, target):
    """Merge source branch into target."""
    memory_dir = ctx.obj["memory_dir"]

    if not memory_dir.exists():
        console.print(f"[red]Memory directory does not exist:[/red] {memory_dir}")
        return

    engine = get_engine(memory_dir)
    target_branch = target or engine.branch_manager.current_branch_name()

    try:
        branch_obj = engine.branch_manager.merge(source, target_branch)
        console.print(f"[green]✓[/green] Merged '{source}' into '{target_branch}'")
        console.print(f"  Entities: {len(branch_obj.entity_ids)}")
        console.print(f"  Relations: {len(branch_obj.relation_ids)}")
    except ValueError as e:
        console.print(f"[red]Error:[/red] {e}")


@branch.command("archive")
@click.argument("name")
@click.pass_context
def branch_archive(ctx, name):
    """Archive a branch."""
    memory_dir = ctx.obj["memory_dir"]

    if not memory_dir.exists():
        console.print(f"[red]Memory directory does not exist:[/red] {memory_dir}")
        return

    engine = get_engine(memory_dir)

    try:
        branch_obj = engine.branch_manager.archive(name)
        console.print(f"[green]✓[/green] Archived branch '{name}' -> '{branch_obj.name}'")
    except ValueError as e:
        console.print(f"[red]Error:[/red] {e}")


@branch.command("unarchive")
@click.argument("name")
@click.option("--new-name", help="New name for restored branch")
@click.pass_context
def branch_unarchive(ctx, name, new_name):
    """Unarchive a branch."""
    memory_dir = ctx.obj["memory_dir"]

    if not memory_dir.exists():
        console.print(f"[red]Memory directory does not exist:[/red] {memory_dir}")
        return

    engine = get_engine(memory_dir)

    try:
        branch_obj = engine.branch_manager.unarchive(name, new_name=new_name)
        console.print(f"[green]✓[/green] Unarchived branch '{name}' -> '{branch_obj.name}'")
    except ValueError as e:
        console.print(f"[red]Error:[/red] {e}")


# ─────────────────────────────────────────────────────────────────────────────
# Visualization Commands
# ─────────────────────────────────────────────────────────────────────────────


@cli.command()
@click.option("--export", "export_path", type=click.Path(path_type=Path), help="Export only (don't open viewer)")
@click.option("--with-context/--no-context", default=True, help="Include off-branch entities as ghost nodes")
@click.option("--open", "open_only", type=click.Path(path_type=Path), help="Open an existing export file")
@click.option("--watch", is_flag=True, help="Keep server running for live refresh")
@click.pass_context
def graph(ctx, export_path, with_context, open_only, watch):
    """Visualize the knowledge graph."""
    import webbrowser
    from .viz import export_graph_for_viz
    from .branches import BranchManager

    memory_dir = ctx.obj["memory_dir"]

    if open_only:
        from .viz import create_standalone_viewer

        export_file = Path(open_only)
        if not export_file.exists():
            console.print(f"[red]✗[/red] File not found: {open_only}")
            return

        data = json.loads(export_file.read_text())
        viewer_path = memory_dir / "viz" / "graph-viewer.html"
        create_standalone_viewer(data, viewer_path)
        url = f"file://{viewer_path.resolve()}"
        webbrowser.open(url)
        console.print(f"[green]✓[/green] Opened viewer with {open_only}")
        return

    event_store = EventStore(memory_dir / "mnemograph.db")
    events = event_store.read_all()
    state = materialize(events)

    if len(state.entities) == 0:
        console.print("[yellow]![/yellow] No entities in graph. Create some first!")
        return

    branch_manager = BranchManager(memory_dir, lambda: state)
    current_branch = branch_manager.current_branch_name()

    if current_branch == "main":
        branch_entity_ids = None
        branch_relation_ids = None
    else:
        branch_obj = branch_manager.get(current_branch)
        if branch_obj:
            branch_entity_ids = branch_obj.entity_ids
            branch_relation_ids = branch_obj.relation_ids
        else:
            branch_entity_ids = None
            branch_relation_ids = None

    output_path = export_graph_for_viz(
        state=state,
        memory_dir=memory_dir,
        branch_entity_ids=branch_entity_ids,
        branch_relation_ids=branch_relation_ids,
        branch_name=current_branch,
        include_context=with_context,
        output_path=export_path,
    )

    console.print(f"[green]✓[/green] Exported to: {output_path}")
    console.print(f"   {len(state.entities)} entities, {len(state.relations)} relations")

    if not export_path:
        from .viz import create_standalone_viewer
        import http.server
        import threading
        import socket

        data = json.loads(output_path.read_text())
        viewer_path = memory_dir / "viz" / "graph-viewer.html"
        create_standalone_viewer(data, viewer_path, api_enabled=watch)

        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(('', 0))
            port = s.getsockname()[1]

        viz_dir = memory_dir / "viz"
        shutdown_requested = threading.Event()

        class GraphAPIHandler(http.server.SimpleHTTPRequestHandler):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, directory=str(viz_dir), **kwargs)

            def log_message(self, format, *args):
                pass

            def do_GET(self):
                if self.path == "/api/graph":
                    event_store_fresh = EventStore(memory_dir / "mnemograph.db")
                    events_fresh = event_store_fresh.read_all()
                    state_fresh = materialize(events_fresh)

                    branch_manager_fresh = BranchManager(memory_dir, lambda: state_fresh)
                    current_branch_fresh = branch_manager_fresh.current_branch_name()

                    if current_branch_fresh == "main":
                        fresh_entity_ids = None
                        fresh_relation_ids = None
                    else:
                        branch_fresh = branch_manager_fresh.get(current_branch_fresh)
                        if branch_fresh:
                            fresh_entity_ids = branch_fresh.entity_ids
                            fresh_relation_ids = branch_fresh.relation_ids
                        else:
                            fresh_entity_ids = None
                            fresh_relation_ids = None

                    fresh_output = export_graph_for_viz(
                        state=state_fresh,
                        memory_dir=memory_dir,
                        branch_entity_ids=fresh_entity_ids,
                        branch_relation_ids=fresh_relation_ids,
                        branch_name=current_branch_fresh,
                        include_context=with_context,
                    )
                    fresh_data = fresh_output.read_text()

                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Access-Control-Allow-Origin", "*")
                    self.end_headers()
                    self.wfile.write(fresh_data.encode())
                else:
                    super().do_GET()

            def do_POST(self):
                if self.path == "/api/shutdown":
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Access-Control-Allow-Origin", "*")
                    self.end_headers()
                    self.wfile.write(b'{"status": "shutting_down"}')
                    shutdown_requested.set()
                else:
                    self.send_response(404)
                    self.end_headers()

        server = http.server.HTTPServer(('localhost', port), GraphAPIHandler)

        if watch:
            url = f"http://localhost:{port}/graph-viewer.html"
            webbrowser.open(url)
            console.print(f"[green]✓[/green] Opened viewer at {url}")
            console.print("[cyan]   Watching for changes. Press Ctrl+C or click 'Stop Server' to stop.[/cyan]")
            console.print("[dim]   Click 'Refresh' in viewer to reload data.[/dim]")

            def serve_loop():
                while not shutdown_requested.is_set():
                    server.handle_request()

            server_thread = threading.Thread(target=serve_loop, daemon=True)
            server_thread.start()

            try:
                while not shutdown_requested.is_set():
                    shutdown_requested.wait(timeout=0.5)
                console.print("\n[yellow]Server stopped via browser.[/yellow]")
            except KeyboardInterrupt:
                console.print("\n[yellow]Server stopped.[/yellow]")
            finally:
                shutdown_requested.set()
                server.shutdown()
        else:
            def serve_requests():
                for _ in range(5):
                    server.handle_request()

            thread = threading.Thread(target=serve_requests, daemon=True)
            thread.start()

            url = f"http://localhost:{port}/graph-viewer.html"
            webbrowser.open(url)
            console.print(f"[green]✓[/green] Opened viewer at {url}")
            console.print("[dim]   (server will stop after a few seconds)[/dim]")

            thread.join(timeout=5)


# ─────────────────────────────────────────────────────────────────────────────
# Legacy entry point for backward compatibility
# ─────────────────────────────────────────────────────────────────────────────


def main():
    """Legacy argparse entry point - redirects to Click CLI."""
    console.print("[yellow]Note:[/yellow] 'mnemograph-cli' is deprecated. Use 'mnemograph' instead.")
    cli()


if __name__ == "__main__":
    cli()
