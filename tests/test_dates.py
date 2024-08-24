from datetime import datetime, date
from zoneinfo import ZoneInfo

import numpy as np

from orcestra.utils import parse_datestr


def test_parse_date():
    refdate = datetime.fromisoformat("2024-01-01")

    assert parse_datestr("2024-01-01") == refdate
    assert parse_datestr(datetime(2024, 1, 1)) == refdate
    assert parse_datestr(date(2024, 1, 1)) == refdate
    assert parse_datestr(date(2024, 1, 1)) == refdate


def test_parse_date_iana():
    refdate = datetime.fromisoformat("2024-01-01 00:00").replace(tzinfo=ZoneInfo("UTC"))
    assert parse_datestr("2024-01-01 00:00Z") == refdate

    refdate = datetime.fromisoformat("2024-01-01 00:00").replace(
        tzinfo=ZoneInfo("Europe/Berlin")
    )
    assert parse_datestr("2024-01-01 00:00[Europe/Berlin]") == refdate


def test_parse_numpy():
    refdate = datetime.fromisoformat("2024-01-01 00:00").replace(tzinfo=ZoneInfo("UTC"))

    assert parse_datestr(np.datetime64("2024-01-01 00:00")) == refdate
