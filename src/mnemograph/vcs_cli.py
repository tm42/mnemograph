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
@click.option(
    "--global", "use_global",
    is_flag=True,
    help="Use global memory (~/.claude/memory) instead of project-local",
)
@click.pass_context
def cli(ctx, memory_path, use_global):
    """Mnemograph - Git-based knowledge graph version control."""
    ctx.ensure_object(dict)
    if use_global:
        ctx.obj["memory_dir"] = Path.home() / ".claude" / "memory"
    else:
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


# --- Graph Coherence Commands ---


@cli.command()
@click.option("--fix", is_flag=True, help="Interactive cleanup mode")
@click.option("--json", "as_json", is_flag=True, help="Output as JSON")
@click.pass_context
def health(ctx, fix, as_json):
    """Show graph health report.

    Detects issues like orphan entities, potential duplicates,
    overloaded entities, and weak relations.

    Examples:
        mg health              # Show health report
        mg health --fix        # Interactive cleanup mode
        mg health --json       # Output as JSON
    """
    memory_dir = ctx.obj["memory_dir"]
    engine = MemoryEngine(memory_dir, session_id="cli")

    report = engine.get_graph_health()

    if as_json:
        console.print(json.dumps(report, indent=2, default=str))
        return

    summary = report["summary"]

    console.print("[bold]Graph Health Report[/bold]")
    console.print("═" * 40)
    console.print(f"Entities: {summary['total_entities']}    Relations: {summary['total_relations']}")
    console.print()

    # Issues summary
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

    # Recommendations
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

    # Handle orphans
    if issues["orphans"]:
        console.print()
        console.print(f"[bold]Found {len(issues['orphans'])} orphan entities:[/bold]")

        for orphan in issues["orphans"][:10]:  # Limit to 10
            console.print()
            console.print(f"  [cyan]{orphan['name']}[/cyan] ({orphan['type']})")
            console.print(f"  {orphan['observation_count']} observations, created {orphan['created_at'][:10]}")

            choice = click.prompt(
                "  [c]onnect  [d]elete  [s]kip",
                type=click.Choice(["c", "d", "s"], case_sensitive=False),
                default="s",
            )

            if choice == "c":
                # Get suggestions
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

    # Handle duplicates
    if issues["potential_duplicates"]:
        console.print()
        console.print(f"[bold]Found {len(issues['potential_duplicates'])} potential duplicate groups:[/bold]")

        for group in issues["potential_duplicates"][:5]:  # Limit to 5
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
    """Find entities similar to a given name.

    Use before creating new entities to check for duplicates.

    Examples:
        mg similar "React"
        mg similar "PostgreSQL" -t 0.8
    """
    memory_dir = ctx.obj["memory_dir"]
    engine = MemoryEngine(memory_dir, session_id="cli")

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
        table.add_row(
            r["name"],
            r["type"],
            sim_display,
            str(r["observation_count"]),
        )

    console.print(table)


@cli.command()
@click.option("--json", "as_json", is_flag=True, help="Output as JSON")
@click.pass_context
def orphans(ctx, as_json):
    """Find entities with no relations.

    Orphan entities are often incomplete — they should be connected,
    merged into another entity, or deleted.

    Examples:
        mg orphans
        mg orphans --json
    """
    memory_dir = ctx.obj["memory_dir"]
    engine = MemoryEngine(memory_dir, session_id="cli")

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
        table.add_row(
            r["name"],
            r["type"],
            str(r["observation_count"]),
            r["created_at"][:10],
        )

    console.print(table)
    console.print()
    console.print("[dim]Run 'mg health --fix' for interactive cleanup.[/dim]")


@cli.command()
@click.argument("entity")
@click.option("-n", "--limit", type=int, default=5, help="Max suggestions")
@click.option("--json", "as_json", is_flag=True, help="Output as JSON")
@click.pass_context
def suggest(ctx, entity, limit, as_json):
    """Suggest potential relations for an entity.

    Based on semantic similarity and co-occurrence in observations.

    Examples:
        mg suggest "FastAPI"
        mg suggest "FastAPI" -n 10
    """
    memory_dir = ctx.obj["memory_dir"]
    engine = MemoryEngine(memory_dir, session_id="cli")

    results = engine.suggest_relations(entity, limit=limit)

    if as_json:
        console.print(json.dumps(results, indent=2, default=str))
        return

    # Check for error
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
        table.add_row(
            r["target"],
            r["suggested_relation"],
            f"{r['confidence']:.0%}",
            r.get("reason", ""),
        )

    console.print(table)


@cli.command()
@click.option("--reason", "-m", help="Reason for clearing (recorded in event)")
@click.option("--yes", "-y", is_flag=True, help="Skip confirmation prompt")
@click.option("--json", "as_json", is_flag=True, help="Output as JSON")
@click.pass_context
def clear(ctx, reason, yes, as_json):
    """Clear all entities and relations from the graph.

    This is event-sourced — you can rewind to before the clear
    using 'mg show --at <timestamp>' or view history with 'mg log'.

    Use sparingly! This is a destructive operation.

    Examples:
        mg clear                           # Interactive confirmation
        mg clear -y                        # Skip confirmation
        mg clear -m "Starting fresh for v2"  # Record reason
    """
    memory_dir = ctx.obj["memory_dir"]
    engine = MemoryEngine(memory_dir, session_id="cli")

    entity_count = len(engine.state.entities)
    relation_count = len(engine.state.relations)

    if entity_count == 0 and relation_count == 0:
        if as_json:
            console.print(json.dumps({"status": "empty", "message": "Graph is already empty"}, indent=2))
        else:
            console.print("[yellow]![/yellow] Graph is already empty. Nothing to clear.")
        return

    # Confirmation prompt
    if not yes:
        console.print(f"[yellow]⚠  This will clear ALL {entity_count} entities and {relation_count} relations.[/yellow]")
        console.print("[dim]   (Event is recorded — can rewind with 'mg show --at <timestamp>')[/dim]")
        console.print()
        if not click.confirm("Are you sure you want to clear the graph?", default=False):
            console.print("[dim]Cancelled.[/dim]")
            return

    # Clear the graph
    result = engine.clear_graph(reason=reason or "")

    if as_json:
        console.print(json.dumps(result, indent=2, default=str))
    else:
        console.print(f"[green]✓[/green] Cleared {result['entities_cleared']} entities, {result['relations_cleared']} relations")
        if reason:
            console.print(f"   Reason: {reason}")
        console.print("[dim]   Tip: Use 'mg show --at <timestamp>' to view graph before clear[/dim]")


# --- Recovery Commands ---


@cli.command()
@click.option("--json", "as_json", is_flag=True, help="Output as JSON")
@click.pass_context
def reload(ctx, as_json):
    """Reload graph state from events.jsonl on disk.

    Use after git operations (checkout, restore) or external edits
    to events.jsonl to sync the CLI with disk state.

    Examples:
        git restore .claude/memory/events.jsonl && mg reload
        mg reload
    """
    memory_dir = ctx.obj["memory_dir"]
    engine = MemoryEngine(memory_dir, session_id="cli")

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
    """Rewind graph to a previous state using git.

    Fast undo — audit trail is in git history only.
    For audit-preserving restore, use 'mg restore' instead.

    Examples:
        mg rewind                          # Undo last commit
        mg rewind --steps 3                # Go back 3 commits
        mg rewind --to-commit abc123       # Restore to specific commit
    """
    memory_dir = ctx.obj["memory_dir"]
    engine = MemoryEngine(memory_dir, session_id="cli")

    if not engine._in_git_repo:
        console.print("[red]Error:[/red] Not in a git repository. Cannot use git-based rewind.")
        console.print("[dim]Tip: Use 'mg restore --to <timestamp>' for event-based restore.[/dim]")
        return

    # Confirmation prompt
    if not yes and not as_json:
        if to_commit:
            console.print(f"[yellow]⚠  This will rewind events.jsonl to commit {to_commit}.[/yellow]")
        else:
            console.print(f"[yellow]⚠  This will rewind events.jsonl by {steps} commit(s).[/yellow]")
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
@click.option("--to", "timestamp", required=True, help="Timestamp to restore to (ISO or relative)")
@click.option("--reason", "-m", help="Reason for restoring (recorded in events)")
@click.option("--yes", "-y", is_flag=True, help="Skip confirmation prompt")
@click.option("--json", "as_json", is_flag=True, help="Output as JSON")
@click.pass_context
def restore(ctx, timestamp, reason, yes, as_json):
    """Restore graph to state at a specific timestamp.

    Event-based restore — full audit trail preserved in events.
    Use 'mg show --at <timestamp>' first to preview the state.

    Examples:
        mg restore --to "2025-01-10T14:00"
        mg restore --to "2 hours ago" --reason "Undoing experiment"
        mg restore --to yesterday
    """
    memory_dir = ctx.obj["memory_dir"]
    engine = MemoryEngine(memory_dir, session_id="cli")

    # Preview the state
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
        console.print("[dim]Tip: Use 'mg show --at <timestamp>' to explore available states.[/dim]")
        return

    # Confirmation prompt
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


@cli.command()
@click.option("--delete-entity", "-e", multiple=True, help="Entity name(s) to delete completely")
@click.option("--reason", "-m", help="Reason for compacting (recorded in events)")
@click.option("--yes", "-y", is_flag=True, help="Skip confirmation prompt")
@click.option("--json", "as_json", is_flag=True, help="Output as JSON")
@click.pass_context
def compact(ctx, delete_entity, reason, yes, as_json):
    """Compact graph: materialize current state, apply deletions, regenerate events.

    This REWRITES the entire event history with a minimal sequence.
    Pre-compaction state is auto-committed to git (if available).

    Use cases:
    - Remove all traces of an entity and its relations
    - Reduce events.jsonl size
    - Clean up after experiments

    CLI only — not available via MCP (user should be in the loop).

    Examples:
        mg compact                                # Just compact, no deletions
        mg compact --delete-entity "test-entity" # Remove specific entity
        mg compact -e "temp1" -e "temp2"         # Remove multiple entities
    """
    memory_dir = ctx.obj["memory_dir"]
    engine = MemoryEngine(memory_dir, session_id="cli")

    current_entities = len(engine.state.entities)
    current_relations = len(engine.state.relations)
    current_events = len(engine.event_store.read_all())

    if current_entities == 0 and current_relations == 0:
        if as_json:
            console.print(json.dumps({"status": "empty", "message": "Graph is already empty"}, indent=2))
        else:
            console.print("[yellow]![/yellow] Graph is already empty. Nothing to compact.")
        return

    # Show what will be deleted
    deleted_info = []
    if delete_entity:
        for name in delete_entity:
            entity_id = engine._resolve_entity(name)
            if entity_id:
                entity = engine.state.entities[entity_id]
                # Count relations involving this entity
                rel_count = sum(
                    1 for r in engine.state.relations
                    if r.from_entity == entity_id or r.to_entity == entity_id
                )
                deleted_info.append(f"  - {entity.name} ({entity.type}) + {rel_count} relations")
            else:
                deleted_info.append(f"  - {name} [dim](not found)[/dim]")

    # Confirmation prompt
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
    import subprocess
    if engine._in_git_repo and engine._git_root:
        events_path = engine.event_store.path
        relative_path = events_path.relative_to(engine._git_root)

        subprocess.run(["git", "add", str(relative_path)], cwd=engine._git_root, capture_output=True)
        commit_msg = f"Pre-compaction snapshot"
        if reason:
            commit_msg += f": {reason}"
        commit_result = subprocess.run(
            ["git", "commit", "-m", commit_msg, "--allow-empty"],
            cwd=engine._git_root,
            capture_output=True,
            text=True,
        )
        git_committed = commit_result.returncode == 0

    # Perform compaction
    from .models import MemoryEvent
    from datetime import datetime, timezone

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
            # Remove from name index
            if name in state._name_to_id:
                del state._name_to_id[name]

            # Remove relations involving this entity
            for rel in list(state.relations):
                if rel.from_entity == entity_id or rel.to_entity == entity_id:
                    state.relations.remove(rel)
                    deleted_relation_count += 1

    # Generate minimal event sequence
    new_events = []

    # Add compact marker event
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

    # Add create_entity events for remaining entities
    for entity in state.entities.values():
        event = MemoryEvent(
            op="create_entity",
            session_id="compact",
            source="user",
            data=entity.model_dump(mode="json"),
        )
        # Preserve original timestamp
        event.ts = entity.created_at
        new_events.append(event)

    # Add create_relation events for remaining relations
    for relation in state.relations:
        event = MemoryEvent(
            op="create_relation",
            session_id="compact",
            source="user",
            data=relation.model_dump(mode="json"),
        )
        # Preserve original timestamp
        event.ts = relation.created_at
        new_events.append(event)

    # Write new events (overwrite file)
    engine.event_store.path.write_text("")  # Clear file
    for event in new_events:
        engine.event_store.append(event)

    # Reload
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


# --- Visualization Commands ---


@cli.command()
@click.option("--export", "export_path", type=click.Path(path_type=Path), help="Export only (don't open viewer)")
@click.option("--with-context/--no-context", default=True, help="Include off-branch entities as ghost nodes")
@click.option("--open", "open_only", type=click.Path(path_type=Path), help="Open an existing export file")
@click.option("--watch", is_flag=True, help="Keep server running for live refresh")
@click.pass_context
def graph(ctx, export_path, with_context, open_only, watch):
    """Visualize the knowledge graph.

    Opens an interactive D3.js viewer in your browser.

    Examples:

        mnemograph graph                    # Export and open viewer

        mnemograph graph --export graph.json  # Export only, no viewer

        mnemograph graph --open exports/graph.json  # Open existing export
    """
    import webbrowser
    from .viz import export_graph_for_viz
    from .events import EventStore
    from .state import materialize

    memory_dir = ctx.obj["memory_dir"]

    # Handle open-only mode
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
        from .viz import create_standalone_viewer
        import http.server
        import threading
        import socket

        # Create standalone HTML with embedded data
        data = json.loads(output_path.read_text())
        viewer_path = memory_dir / "viz" / "graph-viewer.html"
        create_standalone_viewer(data, viewer_path, api_enabled=watch)

        # Find a free port
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(('', 0))
            port = s.getsockname()[1]

        # Start HTTP server
        viz_dir = memory_dir / "viz"

        # Shutdown flag for clean server stop
        shutdown_requested = threading.Event()

        class GraphAPIHandler(http.server.SimpleHTTPRequestHandler):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, directory=str(viz_dir), **kwargs)

            def log_message(self, format, *args):
                pass  # Suppress logging

            def do_GET(self):
                if self.path == "/api/graph":
                    # Re-export fresh data
                    event_store_fresh = EventStore(memory_dir / "events.jsonl")
                    events_fresh = event_store_fresh.read_all()
                    state_fresh = materialize(events_fresh)

                    from .viz import export_graph_for_viz
                    fresh_output = export_graph_for_viz(
                        state=state_fresh,
                        memory_dir=memory_dir,
                        branch_entity_ids=None,
                        branch_relation_ids=None,
                        branch_name="main",
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
                    # Signal shutdown
                    shutdown_requested.set()
                else:
                    self.send_response(404)
                    self.end_headers()

        server = http.server.HTTPServer(('localhost', port), GraphAPIHandler)

        if watch:
            # Watch mode: keep server running
            url = f"http://localhost:{port}/graph-viewer.html"
            webbrowser.open(url)
            console.print(f"[green]✓[/green] Opened viewer at {url}")
            console.print("[cyan]   Watching for changes. Press Ctrl+C or click 'Stop Server' to stop.[/cyan]")
            console.print("[dim]   Click 'Refresh' in viewer to reload data.[/dim]")

            # Run server in a thread so we can check shutdown flag
            def serve_loop():
                while not shutdown_requested.is_set():
                    server.handle_request()

            server_thread = threading.Thread(target=serve_loop, daemon=True)
            server_thread.start()

            try:
                # Wait for either keyboard interrupt or shutdown request
                while not shutdown_requested.is_set():
                    shutdown_requested.wait(timeout=0.5)
                console.print("\n[yellow]Server stopped via browser.[/yellow]")
            except KeyboardInterrupt:
                console.print("\n[yellow]Server stopped.[/yellow]")
            finally:
                shutdown_requested.set()
                server.shutdown()
        else:
            # Normal mode: serve a few requests then stop
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


if __name__ == "__main__":
    cli()
