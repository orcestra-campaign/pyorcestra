from textwrap import dedent
import fsspec
import re
import numpy as np
import xarray as xr
import pandas as pd
from functools import lru_cache

EARTH_CARE_TLES = {
    "sim_operations": dedent(
        """\
        1 99999U 99999A   16000.00000000  .00000000  00000-0  00000-0 0  0000
        2 99999  97.1000 000.0000 0000000   0.0000   0.0000 15.56756757000000
        """
    ),
    # TODO: check inclination and orbital period of calval setup
    #    "sim_calval": dedent("""\
    #        1 99999U 99999A   16000.00000000  .00000000  00000-0  00000-0 0  0000
    #        2 99999  97.1000 000.0000 0000000   0.0000   0.0000 15.56756757000000
    #        """),
}


class TLETrackLoader:
    def __init__(self, tle):
        from sgp4.api import Satrec

        lines = tle.splitlines()
        t = [line for line in lines if line.startswith("1 ")][0]
        s = [line for line in lines if line.startswith("2 ")][0]
        self.satellite = Satrec.twoline2rv(t, s)

    def get_track_for_day(self, day):
        from sgp4.api import SGP4_ERRORS
        from astropy.time import Time
        from astropy.coordinates import (
            TEME,
            CartesianDifferential,
            CartesianRepresentation,
        )
        from astropy import units as u
        from astropy.coordinates import ITRS

        day = np.datetime64(day)
        numpy_time = (
            np.linspace(0, 24 * 60 * 60, 24 * 60 * 6, endpoint=False)
            * np.timedelta64(1000000000, "ns")
            + day
        )
        t = Time(numpy_time)

        error_code, teme_p, teme_v = self.satellite.sgp4_array(
            t.jd1, t.jd2
        )  # in km and km/s
        if np.any(error_code):
            raise RuntimeError(SGP4_ERRORS[np.max(error_code)])

        teme_p = CartesianRepresentation(teme_p.T * u.km)
        teme_v = CartesianDifferential(teme_v.T * u.km / u.s)
        teme = TEME(teme_p.with_differentials(teme_v), obstime=t)

        itrs_geo = teme.transform_to(ITRS(obstime=t))
        location = itrs_geo.earth_location
        return xr.Dataset(
            coords={
                "time": ("time", numpy_time),
                "lon": ("time", location.geodetic.lon.deg),
                "lat": ("time", location.geodetic.lat.deg),
                "alt": ("time", location.geodetic.height.to("m").value),
            }
        )


class CalipsoTrackLoader:
    ten_sec_file_re = re.compile(
        "^.*/CPSO_10second_GT_([0-9]{4})_([0-9]{2})_([0-9]{2}).txt$"
    )

    def __init__(self):
        self.track_files = list(sorted(self.gen_urls()))

    def gen_urls(self):
        fs = fsspec.filesystem("http")
        for info in fs.ls(
            "https://www-calipso.larc.nasa.gov/data/TOOLS/overpass/coords/"
        ):
            if info["type"] == "file":
                if m := self.ten_sec_file_re.match(info["name"]):
                    yield np.datetime64("-".join(m.groups())), info["name"]

    @lru_cache
    def get_track_for_day(self, day):
        day = np.datetime64(day)
        url = None
        for file_start, file_url in self.track_files:
            if file_start > day:
                break
            url = file_url

        df = pd.read_fwf(
            url,
            skiprows=[0, 1, 2, 3, 5],
            widths=[24, 18, 19, 14],
            parse_dates={"time": [0]},
        ).rename(
            {"Latitude (deg)": "lat", "Longitude (deg)": "lon", "Range (km)": "alt"},
            axis=1,
        )
        df["alt"] = df["alt"] * 1000  # convert to meters
        df["time"] = pd.to_datetime(
            df["time"], format="%d %b %Y %H:%M:%S.000", exact=False
        )
        return df.set_index("time").sort_index().to_xarray()
