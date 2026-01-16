"""Time parsing utilities for human-friendly time references.

Supports:
- ISO format: "2025-01-15", "2025-01-15T14:30:00"
- Relative: "7 days ago", "2 weeks ago", "1 month ago"
- Named: "yesterday", "last week", "last month"
"""

import re
from datetime import datetime, timedelta, timezone

from dateutil import parser as dateparser
from dateutil.relativedelta import relativedelta


def parse_time_reference(ref: str, now: datetime | None = None) -> datetime:
    """Parse human-friendly time references.

    Args:
        ref: Time reference string
        now: Reference point for relative times (default: utcnow)

    Returns:
        Parsed datetime (timezone-aware UTC)

    Raises:
        ValueError: If the reference cannot be parsed

    Examples:
        >>> parse_time_reference("2025-01-15")
        datetime(2025, 1, 15, 0, 0, tzinfo=timezone.utc)

        >>> parse_time_reference("7 days ago")  # relative to now
        datetime(...)

        >>> parse_time_reference("yesterday")
        datetime(...)
    """
    if now is None:
        now = datetime.now(timezone.utc)

    ref = ref.strip().lower()

    # Handle named references
    if ref == "yesterday":
        return (now - timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
    if ref == "today":
        return now.replace(hour=0, minute=0, second=0, microsecond=0)
    if ref == "last week":
        return now - timedelta(weeks=1)
    if ref == "last month":
        return now - relativedelta(months=1)
    if ref == "last year":
        return now - relativedelta(years=1)

    # Handle "N units ago" pattern
    ago_match = re.match(r"(\d+)\s*(second|minute|hour|day|week|month|year)s?\s*ago", ref)
    if ago_match:
        amount = int(ago_match.group(1))
        unit = ago_match.group(2)

        if unit == "second":
            return now - timedelta(seconds=amount)
        elif unit == "minute":
            return now - timedelta(minutes=amount)
        elif unit == "hour":
            return now - timedelta(hours=amount)
        elif unit == "day":
            return now - timedelta(days=amount)
        elif unit == "week":
            return now - timedelta(weeks=amount)
        elif unit == "month":
            return now - relativedelta(months=amount)
        elif unit == "year":
            return now - relativedelta(years=amount)

    # Fall back to dateutil parser for ISO and other formats
    try:
        parsed = dateparser.parse(ref)
        if parsed is None:
            raise ValueError(f"Cannot parse time reference: {ref}")

        # Ensure timezone-aware
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)

        return parsed
    except (ValueError, OverflowError, dateparser.ParserError) as e:
        raise ValueError(f"Cannot parse time reference: {ref}") from e


def format_relative_time(dt: datetime, now: datetime | None = None) -> str:
    """Format a datetime as a human-readable relative string.

    Args:
        dt: The datetime to format
        now: Reference point (default: utcnow)

    Returns:
        Human-readable string like "2 days ago", "3 weeks ago"
    """
    from .constants import (
        SECONDS_PER_MINUTE,
        SECONDS_PER_HOUR,
        SECONDS_PER_DAY,
        SECONDS_PER_WEEK,
        SECONDS_PER_MONTH,
        SECONDS_PER_YEAR,
    )

    if now is None:
        now = datetime.now(timezone.utc)

    diff = now - dt

    if diff.total_seconds() < 0:
        return "in the future"

    seconds = int(diff.total_seconds())

    if seconds < SECONDS_PER_MINUTE:
        return f"{seconds} seconds ago"
    elif seconds < SECONDS_PER_HOUR:
        minutes = seconds // SECONDS_PER_MINUTE
        return f"{minutes} minute{'s' if minutes != 1 else ''} ago"
    elif seconds < SECONDS_PER_DAY:
        hours = seconds // SECONDS_PER_HOUR
        return f"{hours} hour{'s' if hours != 1 else ''} ago"
    elif seconds < SECONDS_PER_WEEK:
        days = seconds // SECONDS_PER_DAY
        return f"{days} day{'s' if days != 1 else ''} ago"
    elif seconds < SECONDS_PER_MONTH:
        weeks = seconds // SECONDS_PER_WEEK
        return f"{weeks} week{'s' if weeks != 1 else ''} ago"
    elif seconds < SECONDS_PER_YEAR:
        months = seconds // SECONDS_PER_MONTH
        return f"{months} month{'s' if months != 1 else ''} ago"
    else:
        years = seconds // SECONDS_PER_YEAR
        return f"{years} year{'s' if years != 1 else ''} ago"
