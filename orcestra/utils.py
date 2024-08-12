import datetime
import re
import zoneinfo


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
    # Parse ISO 8601 string and time zone information
    regex = re.compile(r"^(.*?)(?:\[(.*)\])?$")
    iso, tz = regex.match(datestr).groups()

    # Create datetime object and attach timezone info, if available
    date = datetime.datetime.fromisoformat(iso)
    if tz is not None:
        return date.replace(tzinfo=zoneinfo.ZoneInfo(tz))
    else:
        return date
