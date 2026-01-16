"""Linearize graph subgraphs to prose for human-readable recall.

Transforms structured graph data into prose that agents can read directly
without mental parsing of JSON structures.
"""

from __future__ import annotations

from .models import Entity
from .state import GraphState


def linearize_to_prose(
    state: GraphState,
    entities: list[Entity],
    entity_ids: set[str],
    depth: str = "medium",
) -> str:
    """Convert a subgraph to readable prose.

    Groups content by semantic category (projects, decisions, gotchas, questions)
    rather than entity-by-entity listing.

    Args:
        state: The full graph state (for relation lookups)
        entities: The entities to linearize
        entity_ids: Set of entity IDs in the subgraph (for relation filtering)
        depth: Controls verbosity - 'shallow', 'medium', or 'deep'

    Returns:
        Formatted prose string
    """
    if not entities:
        return "No matching entities found."

    sections: list[str] = []

    # Group by type
    by_type = _group_by_type(entities)

    # 1. Projects/services first (most important context)
    if "project" in by_type:
        sections.append(_format_projects(by_type["project"], state, entity_ids, depth))

    # 2. Concepts and patterns
    concepts = by_type.get("concept", []) + by_type.get("pattern", [])
    if concepts:
        sections.append(_format_concepts(concepts, state, entity_ids, depth))

    # 3. Decisions (with rationale)
    if "decision" in by_type:
        sections.append(_format_decisions(by_type["decision"], depth))

    # 4. Gotchas (extracted from observations across all entities)
    gotchas = _extract_gotchas(entities)
    if gotchas:
        sections.append(_format_gotchas(gotchas))

    # 5. Open questions
    if "question" in by_type:
        sections.append(_format_questions(by_type["question"]))

    # 6. Learnings
    if "learning" in by_type:
        sections.append(_format_learnings(by_type["learning"], depth))

    # 7. Generic entities (catch-all)
    if "entity" in by_type:
        sections.append(_format_generic(by_type["entity"], depth))

    return "\n\n".join(s for s in sections if s.strip())


def _group_by_type(entities: list[Entity]) -> dict[str, list[Entity]]:
    """Group entities by their type."""
    by_type: dict[str, list[Entity]] = {}
    for entity in entities:
        etype = entity.type
        if etype not in by_type:
            by_type[etype] = []
        by_type[etype].append(entity)
    return by_type


def _format_projects(
    projects: list[Entity],
    state: GraphState,
    entity_ids: set[str],
    depth: str,
) -> str:
    """Format project entities with their relations."""
    lines: list[str] = []

    for project in projects:
        lines.append(f"**{project.name}** (project)")

        # Key observations (limit based on depth)
        obs_limit = 2 if depth == "shallow" else 4 if depth == "medium" else 10
        obs_texts = [o.text for o in project.observations[:obs_limit]]
        if obs_texts:
            lines.append(". ".join(obs_texts) + ".")

        # Relations — group by type
        uses = _get_relations_by_type(project.id, state, entity_ids, "uses")
        implements = _get_relations_by_type(project.id, state, entity_ids, "implements")
        depends_on = _get_relations_by_type(project.id, state, entity_ids, "depends_on")

        if uses:
            lines.append(f"Uses: {', '.join(uses)}")
        if implements:
            lines.append(f"Implements: {', '.join(implements)}")
        if depends_on:
            lines.append(f"Depends on: {', '.join(depends_on)}")

        lines.append("")

    return "\n".join(lines).strip()


def _format_concepts(
    concepts: list[Entity],
    state: GraphState,
    entity_ids: set[str],
    depth: str,
) -> str:
    """Format concept/pattern entities."""
    if depth == "shallow":
        # Just names
        names = [f"**{c.name}**" for c in concepts[:5]]
        return "Concepts: " + ", ".join(names)

    lines: list[str] = []
    for concept in concepts:
        obs_limit = 2 if depth == "medium" else 5
        obs_texts = [o.text for o in concept.observations[:obs_limit]]
        obs_str = ". ".join(obs_texts) + "." if obs_texts else ""
        lines.append(f"**{concept.name}** ({concept.type}): {obs_str}")

    return "\n".join(lines)


def _format_decisions(
    decisions: list[Entity],
    depth: str,
) -> str:
    """Format decision entities with rationale."""
    lines: list[str] = ["**Decisions:**"]

    for decision in decisions:
        name = decision.name
        # Use first observation as rationale
        rationale = decision.observations[0].text if decision.observations else ""

        if depth == "shallow":
            lines.append(f"• {name}")
        else:
            lines.append(f"• {name} — {rationale}")

    return "\n".join(lines)


def _extract_gotchas(entities: list[Entity]) -> list[str]:
    """Extract gotcha observations from all entities.

    Looks for observations starting with:
    - "Gotcha:"
    - "Warning:"
    - "Caution:"
    - "Note:"
    - "Important:"
    """
    prefixes = ("gotcha:", "warning:", "caution:", "note:", "important:")
    gotchas: list[str] = []

    for entity in entities:
        for obs in entity.observations:
            text_lower = obs.text.lower()
            for prefix in prefixes:
                if text_lower.startswith(prefix):
                    # Remove the prefix and clean up
                    clean_text = obs.text[len(prefix):].strip()
                    gotchas.append(clean_text)
                    break

    return gotchas


def _format_gotchas(gotchas: list[str]) -> str:
    """Format gotcha observations."""
    lines: list[str] = ["**Gotchas:**"]
    for gotcha in gotchas[:10]:  # Limit to 10
        lines.append(f"• {gotcha}")
    return "\n".join(lines)


def _format_questions(questions: list[Entity]) -> str:
    """Format open questions."""
    lines: list[str] = ["**Open questions:**"]
    for question in questions:
        lines.append(f"• {question.name}")
        if question.observations:
            lines.append(f"  ({question.observations[0].text})")
    return "\n".join(lines)


def _format_learnings(learnings: list[Entity], depth: str) -> str:
    """Format learning entities."""
    lines: list[str] = ["**Learnings:**"]

    for learning in learnings:
        if depth == "shallow":
            lines.append(f"• {learning.name}")
        else:
            obs_texts = [o.text for o in learning.observations[:2]]
            detail = " — " + ". ".join(obs_texts) if obs_texts else ""
            lines.append(f"• {learning.name}{detail}")

    return "\n".join(lines)


def _format_generic(
    entities: list[Entity],
    depth: str,
) -> str:
    """Format generic 'entity' type items."""
    lines: list[str] = ["**Other:**"]

    for entity in entities:
        if depth == "shallow":
            lines.append(f"• {entity.name}")
        else:
            obs_limit = 2 if depth == "medium" else 5
            obs_texts = [o.text for o in entity.observations[:obs_limit]]
            detail = ": " + ". ".join(obs_texts) if obs_texts else ""
            lines.append(f"• {entity.name}{detail}")

    return "\n".join(lines)


def _get_relations_by_type(
    entity_id: str,
    state: GraphState,
    entity_ids: set[str],
    relation_type: str,
) -> list[str]:
    """Get target entity names for outgoing relations of a specific type.

    Uses O(k) index lookup where k = edges from entity, instead of O(n) full scan.
    """
    targets: list[str] = []

    for rel in state.get_outgoing_relations(entity_id):
        if rel.type == relation_type and rel.to_entity in entity_ids:
            target = state.entities.get(rel.to_entity)
            if target:
                targets.append(target.name)

    return targets


def linearize_shallow_summary(state: GraphState) -> str:
    """Generate a quick prose summary of the graph.

    For shallow depth — just stats and highlights, no full entity listing.
    """
    entity_count = len(state.entities)
    relation_count = len(state.relations)

    if entity_count == 0:
        return "Memory is empty. No entities stored yet."

    # Type breakdown
    type_counts: dict[str, int] = {}
    for e in state.entities.values():
        type_counts[e.type] = type_counts.get(e.type, 0) + 1

    type_summary = ", ".join(f"{count} {etype}s" for etype, count in sorted(type_counts.items(), key=lambda x: -x[1]))

    # Recent entities
    recent = sorted(state.entities.values(), key=lambda e: e.updated_at, reverse=True)[:5]
    recent_names = [e.name for e in recent]

    # Hot entities (most accessed)
    hot = sorted(state.entities.values(), key=lambda e: e.access_count, reverse=True)[:3]
    hot_names = [e.name for e in hot if e.access_count > 0]

    lines = [
        f"**Memory:** {entity_count} entities, {relation_count} relations ({type_summary})",
        "",
        f"**Recent:** {', '.join(recent_names)}",
    ]

    if hot_names:
        lines.append(f"**Frequently accessed:** {', '.join(hot_names)}")

    # Extract any gotchas from the graph
    all_gotchas = _extract_gotchas(list(state.entities.values()))
    if all_gotchas:
        lines.append("")
        lines.append(f"**Gotchas ({len(all_gotchas)}):** " + all_gotchas[0])
        if len(all_gotchas) > 1:
            lines.append(f"  ... and {len(all_gotchas) - 1} more")

    # Open questions count
    questions = [e for e in state.entities.values() if e.type == "question"]
    if questions:
        lines.append(f"**Open questions:** {len(questions)}")

    return "\n".join(lines)
