import datetime
import pathlib
import re
import zoneinfo
from functools import lru_cache

import yaml


def parse_datestr(datestr):
    """Parse an ISO 8601 date string that may include a IANA time zone.

    Examples:
        The following call will assume local time
        >>> parse_datestr("2024-08-01 09:00:00")
        datetime.datetime(2024, 8, 1, 9, 0)

        Date and time in UTC
        >>> parse_datestr("2024-08-01 09:00:00Z")
        datetime.datetime(2024, 8, 1, 9, 0, tzinfo=datetime.timezone.utc)

        Here, the IANA time zone for Cape Verde Islands will be added
        >>> parse_datestr("2024-08-01 09:00:00[Atlantic/Cape_Verde]")
        datetime.datetime(2024, 8, 1, 9, 0, tzinfo=zoneinfo.ZoneInfo(key='Atlantic/Cape_Verde'))

    """
    if isinstance(datestr, datetime.datetime):
        return datestr
    elif isinstance(datestr, datetime.date):
        return datetime.datetime.combine(datestr, datetime.time())

    # Parse ISO 8601 string and time zone information
    regex = re.compile(r"^(.*?)(?:\[(.*)\])?$")
    iso, tz = regex.match(datestr).groups()

    # Create datetime object and attach timezone info, if available
    date = datetime.datetime.fromisoformat(iso)
    if tz is not None:
        return date.replace(tzinfo=zoneinfo.ZoneInfo(tz))
    else:
        return date


@lru_cache
def load_frontmatter(path):
    """Load and return the front matter section of a YAML file."""
    with pathlib.Path(path).open("r") as fp:
        frontmatter = next(yaml.safe_load_all(fp))

    frontmatter["filepath"] = pathlib.Path(path).as_posix()

    for key in ("takeoff", "landing"):
        if key in frontmatter:
            frontmatter[key] = parse_datestr(frontmatter[key])

    return frontmatter
