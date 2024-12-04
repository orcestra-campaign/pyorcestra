import datetime
import pathlib
import re
import zoneinfo
from functools import lru_cache

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import yaml
import fsspec


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

        In case of numpy datetime64, we assume it's in UTC (that's what most np.datetime64 are)
        >>> parse_datestr(np.datetime64("2024-08-01 09:00:00"))
        datetime.datetime(2024, 8, 1, 9, 0, tzinfo=datetime.timezone.utc)

    """
    if isinstance(datestr, datetime.datetime):
        return datestr
    elif isinstance(datestr, datetime.date):
        return datetime.datetime.combine(datestr, datetime.time())
    elif isinstance(datestr, xr.DataArray):
        return (
            pd.Timestamp(datestr.values)
            .to_pydatetime(warn=False)
            .replace(tzinfo=datetime.timezone.utc)
        )
    elif isinstance(datestr, np.datetime64):
        return (
            pd.Timestamp(datestr)
            .to_pydatetime(warn=False)
            .replace(tzinfo=datetime.timezone.utc)
        )

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
    with pathlib.Path(path).open("r", encoding="utf8") as fp:
        frontmatter = next(yaml.safe_load_all(fp))

    frontmatter["filepath"] = pathlib.Path(path).as_posix()

    for key in ("takeoff", "landing"):
        if key in frontmatter:
            frontmatter[key] = parse_datestr(frontmatter[key])

    return frontmatter


def export_planet(fname, fig=None, dpi=144, stem_ext="_planet", **kwargs):
    """Save a Matplotlib figure in high and low resolution for PLANET upload.

    This function is intended as a drop-in replacement for Matplotlib's `savefig()`.
    It will save the image based on the format extension and settings passed by the user.
    It will also save a low-res JPEG with an extension to the filename stem (default "_planet").
    """
    if fig is None:
        fig = plt.gcf()

    # Store "as is" for people on ground
    fig.savefig(fname, dpi=dpi, **kwargs)

    # Low-res version for PLANET upload
    fpath = pathlib.Path(fname)
    fig.savefig(
        fname=fpath.with_stem(fpath.stem + stem_ext).with_suffix(".jpeg"),
        dpi=72,
        pil_kwargs={
            "optimize": True,
            "progressive": True,
        },
        **kwargs,
    )


@lru_cache
def get_flight_segments():
    flight_segment_file = (
        "https://orcestra-campaign.github.io/flight_segmentation/all_flights.yaml"
    )
    with fsspec.open(flight_segment_file) as f:
        meta = yaml.safe_load(f)
    return meta
