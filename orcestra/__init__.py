from .flightplan import LatLon, bco, sal
from .utils import parse_datestr, get_flight_segments
from .io import read_igi, read_bahamas_100hz

import warnings

# intake_xarray will have a bug and eventually needs an update >= 0.4.1
# while this update is merged in intake_xarray main, it's not yet released
# for now, we just silence the warning, as the current implementation still works
warnings.filterwarnings(
    action="ignore",
    message="",
    category=FutureWarning,
    module="intake_xarray",
)

__all__ = [
    "LatLon",
    "bco",
    "sal",
    "parse_datestr",
    "get_flight_segments",
    "read_igi",
    "read_bahamas_100hz",
]
