"""CLI for claude-memory version control."""

import os
from pathlib import Path

import click
from rich.console import Console

from .vcs import MemoryVCS

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


if __name__ == "__main__":
    cli()
