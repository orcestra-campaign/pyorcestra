import getpass
import json
import pathlib
from datetime import datetime
from io import BytesIO

import numpy as np
import pandas as pd
import requests


def get_planet_credentials():
    """Get credentials for PLANET authentication.

    User name and password can be stored in a file ~/.config/orcestra/planet.json as:

    {
        "user": <username>,
        "password": <password>
    }

    Otherwise, the user is prompted for input.
    """
    passwd_file = pathlib.Path("~/.config/orcestra/planet.json").expanduser().resolve()

    if passwd_file.is_file():
        with passwd_file.open("r") as fp:
            j = json.loads(fp.read())

            return j["user"], j["password"]

    return input("Username: "), getpass.getpass("Password: ")


def request_telemetry(user, password, mission_id="PERCUSION", vehicle_id="METEOR"):
    """Retrieve telemetry information from the PLANET system."""

    # Retrieve METEOR position via PLANET API
    url = f"https://dlr.atmosphere.aero/api/v1/telemetry/{mission_id}/"
    query_parameters = {
        "creator": vehicle_id,
        "fmt": "csv",
    }

    r = requests.get(url, params=query_parameters, auth=(user, password))
    r.raise_for_status()

    return r


def get_meteor_track(deduplicate_latlon=False):
    """Request the METEOR track from PLANET and return as xr.Dataset.

    Args:
      deduplicate_latlon (bool): Remove duplicated lat/lon pairs
        for, e.g., faster plotting (default is `False`).
    """
    # Retrieve telemetry data for METEOR
    user, password = get_planet_credentials()
    r = request_telemetry(user, password)

    # Read CSV response as pandas dataframe
    ds = pd.read_csv(BytesIO(r.text.encode())).to_xarray()

    # Sanitise dataset
    ds = (
        ds[["issued", "position.0", "position.1"]]
        .drop_vars(
            "index",
        )
        .rename(
            {
                "index": "time",
                "issued": "time",
                "position.0": "lon",
                "position.1": "lat",
            }
        )
    )

    # Properly parse datetimes
    ds = ds.assign_coords(time=[datetime.fromisoformat(str(t)) for t in ds.time.values])

    if deduplicate_latlon:
        # Remove duplicate lat/lon information
        pos = [(lon, lat) for lon, lat in zip(ds.lon.values, ds.lat.values)]
        idx = np.sum(np.diff(pos, axis=0, prepend=1000), axis=1).astype(bool)
        ds = ds.isel(time=idx).sortby("time")

    return ds
