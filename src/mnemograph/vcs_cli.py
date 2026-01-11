"""CLI for claude-memory version control.

Includes git-based versioning and event rewind capabilities.
"""

import json
import os
from datetime import datetime, timezone
from pathlib import Path

import click
from rich.console import Console
from rich.table import Table

from .vcs import MemoryVCS
from .engine import MemoryEngine
from .timeutil import parse_time_reference, format_relative_time

console = Console()


def get_memory_dir() -> Path:
    """Find memory directory from MEMORY_PATH env or walk up to find .claude/memory."""
    # Check environment variable first
    if env_path := os.environ.get("MEMORY_PATH"):
        return Path(env_path)

    # Walk up to find .claude/memory
    cwd = Path.cwd()
    for parent in [cwd] + list(cwd.parents):
        memory_dir = parent / ".claude" / "memory"
        if memory_dir.exists():
            return memory_dir

    # Default to cwd/.claude/memory
    return cwd / ".claude" / "memory"


@click.group()
@click.option(
    "--memory-path",
    envvar="MEMORY_PATH",
    type=click.Path(path_type=Path),
    help="Path to memory directory",
)
@click.pass_context
def cli(ctx, memory_path):
    """Claude Memory - Git-based knowledge graph version control."""
    ctx.ensure_object(dict)
    ctx.obj["memory_dir"] = memory_path or get_memory_dir()


@cli.command()
@click.pass_context
def init(ctx):
    """Initialize memory repository."""
    vcs = MemoryVCS(ctx.obj["memory_dir"])
    if vcs.init():
        console.print(f"[green]✓[/green] Initialized memory repository at {ctx.obj['memory_dir']}")
    else:
        console.print(f"[yellow]![/yellow] Repository already initialized at {ctx.obj['memory_dir']}")


@cli.command()
@click.pass_context
def status(ctx):
    """Show working tree status."""
    try:
        vcs = MemoryVCS(ctx.obj["memory_dir"])
        st = vcs.status()
    except RuntimeError as e:
        console.print(f"[red]Error:[/red] {e}")
        return

    console.print(f"On branch: [cyan]{st['branch']}[/cyan]")
    if st["commit"]:
        console.print(f"Last commit: [yellow]{st['commit']}[/yellow] {st['commit_message']}")

    console.print()

    # Get graph stats from current state
    from .events import EventStore
    from .state import materialize

    event_store = EventStore(ctx.obj["memory_dir"] / "events.jsonl")
    events = event_store.read_all()
    state = materialize(events)

    console.print(f"Graph state: [bold]{len(state.entities)}[/bold] entities, [bold]{len(state.relations)}[/bold] relations")
    console.print()

    if st["uncommitted_events"]:
        console.print(f"[yellow]Uncommitted events: {len(st['uncommitted_events'])}[/yellow]")
        for event in st["uncommitted_events"][:10]:
            op = event.get("op", "?")
            if op == "create_entity":
                name = event.get("data", {}).get("name", "?")
                etype = event.get("data", {}).get("type", "entity")
                console.print(f"  [green]+[/green] {op}: \"{name}\" ({etype})")
            elif op == "create_relation":
                data = event.get("data", {})
                console.print(f"  [green]+[/green] {op}: {data.get('from_entity', '?')} -> {data.get('to_entity', '?')}")
            elif op == "add_observation":
                entity_name = event.get("data", {}).get("entity_name", "?")
                console.print(f"  [blue]~[/blue] {op}: {entity_name}")
            else:
                console.print(f"  [blue]~[/blue] {op}")

        if len(st["uncommitted_events"]) > 10:
            console.print(f"  ... and {len(st['uncommitted_events']) - 10} more")
    else:
        console.print("[green]Nothing to commit, working tree clean[/green]")


@cli.command()
@click.option("-m", "--message", required=True, help="Commit message")
@click.option("-a", "--auto-summary", is_flag=True, help="Add auto-generated summary")
@click.pass_context
def commit(ctx, message, auto_summary):
    """Commit changes to events."""
    try:
        vcs = MemoryVCS(ctx.obj["memory_dir"])
        commit_hash = vcs.commit(message, auto_summary=auto_summary)
        console.print(f"[green]✓[/green] [{commit_hash}] {message}")
    except RuntimeError as e:
        console.print(f"[red]Error:[/red] {e}")


@cli.command()
@click.option("-n", "--max-count", default=10, help="Number of commits to show")
@click.option("--oneline", is_flag=True, help="Show one line per commit")
@click.pass_context
def log(ctx, max_count, oneline):
    """Show commit logs."""
    try:
        vcs = MemoryVCS(ctx.obj["memory_dir"])
        commits = vcs.log(n=max_count)
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


# --- Event Rewind Commands ---


@cli.command()
@click.option("--at", "timestamp", help="Show state at this time (ISO, relative, or named)")
@click.option("--ago", help="Show state N units ago (e.g., '7 days', '2 weeks')")
@click.option("--json", "as_json", is_flag=True, help="Output as JSON")
@click.pass_context
def show(ctx, timestamp, ago, as_json):
    """Show graph state at a point in time.

    Examples:
        claude-mem show --at "2025-01-15"
        claude-mem show --at "yesterday"
        claude-mem show --ago "7 days"
    """
    memory_dir = ctx.obj["memory_dir"]

    # Determine timestamp
    if ago:
        ts_str = f"{ago} ago"
    elif timestamp:
        ts_str = timestamp
    else:
        # Default to current state
        ts_str = None

    try:
        engine = MemoryEngine(memory_dir, session_id="cli")

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
                    table.add_row(
                        entity.name,
                        entity.type,
                        str(len(entity.observations)),
                    )

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
    """Show what changed between two points in time.

    Examples:
        claude-mem diff "2025-01-01"
        claude-mem diff "last week" --to "yesterday"
        claude-mem diff "7 days ago"
    """
    memory_dir = ctx.obj["memory_dir"]

    try:
        engine = MemoryEngine(memory_dir, session_id="cli")
        result = engine.diff_between(start, end)

        if as_json:
            console.print(json.dumps(result, indent=2))
        else:
            console.print(f"[bold]Changes from {result['start_timestamp']} to {result['end_timestamp']}[/bold]")
            console.print(f"Events in range: {result['event_count']}")
            console.print()

            entities = result["entities"]
            relations = result["relations"]

            # Entities
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

            # Relations
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
    """Show the history of changes to an entity.

    Examples:
        claude-mem history "Mnemograph"
    """
    memory_dir = ctx.obj["memory_dir"]

    try:
        engine = MemoryEngine(memory_dir, session_id="cli")
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


# --- Edge Weight Commands ---


@cli.group()
def relation():
    """Commands for relation weights and connections."""
    pass


@relation.command("show")
@click.argument("relation_id")
@click.option("--json", "as_json", is_flag=True, help="Output as JSON")
@click.pass_context
def relation_show(ctx, relation_id, as_json):
    """Show relation details with weight breakdown.

    Examples:
        claude-mem relation show 01J...
    """
    memory_dir = ctx.obj["memory_dir"]
    engine = MemoryEngine(memory_dir, session_id="cli")

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
    """Set explicit importance weight for a relation.

    Weight should be between 0.0 (unimportant) and 1.0 (critical).

    Examples:
        claude-mem relation set-weight 01J... 0.9
    """
    memory_dir = ctx.obj["memory_dir"]
    engine = MemoryEngine(memory_dir, session_id="cli")

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
    """List weak relations (candidates for pruning).

    Examples:
        claude-mem relation list --max-weight 0.2
        claude-mem relation list -n 50
    """
    memory_dir = ctx.obj["memory_dir"]
    engine = MemoryEngine(memory_dir, session_id="cli")

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
            table.add_row(
                r["from_entity"][:20],
                r["relation_type"],
                r["to_entity"][:20],
                f"{r['weight']:.3f}",
            )

        console.print(table)
        console.print()
        console.print(f"[dim]Found {len(results)} weak relation(s)[/dim]")


@cli.command()
@click.argument("entity_name")
@click.option("-n", "--limit", type=int, default=10, help="Max connections to show")
@click.option("--json", "as_json", is_flag=True, help="Output as JSON")
@click.pass_context
def connections(ctx, entity_name, limit, as_json):
    """Show an entity's strongest connections by weight.

    Examples:
        claude-mem connections "Mnemograph"
        claude-mem connections "Mnemograph" -n 20
    """
    memory_dir = ctx.obj["memory_dir"]
    engine = MemoryEngine(memory_dir, session_id="cli")

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


# --- Visualization Commands ---


@cli.command()
@click.option("--export", "export_path", type=click.Path(path_type=Path), help="Export only (don't open viewer)")
@click.option("--with-context/--no-context", default=True, help="Include off-branch entities as ghost nodes")
@click.option("--open", "open_only", type=click.Path(path_type=Path), help="Open an existing export file")
@click.pass_context
def graph(ctx, export_path, with_context, open_only):
    """Visualize the knowledge graph.

    Opens an interactive D3.js viewer in your browser.

    Examples:

        mnemograph graph                    # Export and open viewer

        mnemograph graph --export graph.json  # Export only, no viewer

        mnemograph graph --open exports/graph.json  # Open existing export
    """
    import webbrowser
    from .viz import export_graph_for_viz, ensure_viewer_exists
    from .events import EventStore
    from .state import materialize

    memory_dir = ctx.obj["memory_dir"]

    # Handle open-only mode
    if open_only:
        viewer_path = ensure_viewer_exists(memory_dir)
        export_abs = Path(open_only).resolve()
        url = f"file://{viewer_path}#{export_abs}"
        webbrowser.open(url)
        console.print(f"[green]✓[/green] Opened viewer with {open_only}")
        return

    # Load current state
    event_store = EventStore(memory_dir / "events.jsonl")
    events = event_store.read_all()
    state = materialize(events)

    if len(state.entities) == 0:
        console.print("[yellow]![/yellow] No entities in graph. Create some first!")
        return

    # Export graph
    output_path = export_graph_for_viz(
        state=state,
        memory_dir=memory_dir,
        branch_entity_ids=None,  # TODO: branch support
        branch_relation_ids=None,
        branch_name="main",
        include_context=with_context,
        output_path=export_path,
    )

    console.print(f"[green]✓[/green] Exported to: {output_path}")
    console.print(f"   {len(state.entities)} entities, {len(state.relations)} relations")

    # Open viewer unless export-only
    if not export_path:
        viewer_path = ensure_viewer_exists(memory_dir)
        output_abs = output_path.resolve()
        url = f"file://{viewer_path}#{output_abs}"
        webbrowser.open(url)
        console.print(f"[green]✓[/green] Opened viewer in browser")


if __name__ == "__main__":
    cli()
