import warnings
import fsspec
import requests
import re
import numpy as np
import os
import xarray as xr
import pandas as pd
from datetime import datetime, timedelta, timezone
from functools import lru_cache
from io import StringIO, BytesIO
from PIL import Image


def get_tle(**query):
    """
    GET TLE from celestrak.org. Query must be key-value arguments according to
    https://celestrak.org/NORAD/documentation/gp-data-formats.php
    e.g. NAME="EARTHCARE"
    """
    res = requests.get("https://celestrak.org/NORAD/elements/gp.php", params=query)
    res.raise_for_status()
    return res.text


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


def earthcare_track_loader():
    return TLETrackLoader(get_tle(NAME="EARTHCARE"))


def pace_track_loader():
    return TLETrackLoader(get_tle(CATNR="58928"))


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


class SattrackLoader:
    server = "https://sattracks.orcestra-campaign.org/"

    def __init__(self, satelite_name, forecast_day, kind="LTP", roi="CAPE_VERDE"):
        """
        Loader for satallite tracks from sattracks.orcestra-campaign.org.

        Currently it might be difficult to get valid values for the
        required parameters. If in doubt, check
        https://github.com/orcestra-campaign/sattracks.


        Parameters
        ----------
        satelite_name : str
            name of the satellite to track (e.g. "EARTHCARE")
        forecast_day : np.datetime64 | str
            date on which the forecast for the track has been issued
        """
        self.satelite_name = satelite_name
        self.forecast_day = pd.Timestamp(np.datetime64(forecast_day))
        self.kind = kind
        self.roi = roi

    @lru_cache
    def _get_index(self):
        return (
            pd.read_csv(
                self.server + "index_v2.csv", parse_dates=["forecast_day", "valid_day"]
            )
            .set_index(["roi", "sat", "valid_day", "forecast_day", "kind"])
            .sort_index()
            .loc[self.roi]
        )

    @lru_cache
    def get_track_for_day(self, day):
        """
        Get track data for a given day

        Parameters
        ----------
        forecast_day : np.datetime64 | str
            date for which the prediction is requested

        Returns
        -------
        xr.Dataset
            Dataset containing forcasted satellite track.
        """
        valid_day = pd.Timestamp(np.datetime64(day))
        index_for_day = self._get_index().loc[(self.satelite_name, valid_day)]

        if (
            self.forecast_day in index_for_day.index
            and self.forecast_day < index_for_day.index.max()[0]
        ):
            warnings.warn(
                f"You are using an old forecast (issued on {self.forecast_day.date()}) for {self.satelite_name} on {valid_day.date()}! The newest forecast issued so far was issued on {index_for_day.index.max()[0].date()}. It's a {index_for_day.index.max()[1]} forecast."
            )

        url = (
            self.server
            + index_for_day.loc[self.forecast_day, self.kind]["forecast_file"]
        )
        res = requests.get(url)
        res.raise_for_status()
        text = res.text
        df = pd.read_csv(
            StringIO(text),
            sep=r"\s+",
            header=1,
            date_format="%H:%M:%S",
            parse_dates=["GMT"],
            na_values=[-999],
        )
        df["GMT"] = (
            df["GMT"]
            - pd.to_datetime("00:00:00", format="%H:%M:%S")
            + pd.to_datetime(text.split("\n")[0].split()[0].strip(), format="%Y/%m/%d")
        )

        df = df.rename(
            {
                "GMT": "time",
                "SUBLAT[W+]": "lat",
                "SUBLON[N+]": "lon",
                "SUBLAT[N+]": "lat",
                "SUBLON[W+]": "lon",
                "HEADING": "heading",
            },
            axis=1,
        )
        df["lon"] = -df["lon"]  # east positive
        return df.loc[df.lat.notna()].set_index("time").to_xarray()


@lru_cache
def request_goes(
    time: str,
    layer: str = "GOES-East_ABI_GeoColor",
    extent: tuple = (-60, -40, 5, 20),
    retry: bool = True,
):
    """
    Downloads a GOES snapshot from NASA Worldview and saves it to a file.

    :param time: The timestamp in ISO format (e.g., '2024-09-19T06:00:00Z').
    :param layer: The layer type.
    :param extent: Extent of the snapshot.
    :param retry: If the returned image is pitch black, retry at an ealier timestep.
    """
    bbox = f"{extent[2]},{extent[0]},{extent[3]},{extent[1]}"
    url = f"https://wvs.earthdata.nasa.gov/api/v1/snapshot?REQUEST=GetSnapshot&TIME={time}&BBOX={bbox}&CRS=EPSG:4326&LAYERS={layer},Coastlines_15m&WRAP=x,x&FORMAT=image/jpeg&WIDTH=876&HEIGHT=652&ts=1725710721315"

    response = requests.get(url)

    img = Image.open(BytesIO(response.content))

    while (img.entropy() < 2) and retry:
        time = datetime.fromisoformat(time) - timedelta(minutes=10)
        img = request_goes(
            time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            layer=layer,
            extent=extent,
            retry=retry,
        )

    return img


def goes_snapshot(
    time: str = None,
    layer_type: str = "vis",
    extent: tuple = (-60, -40, 5, 20),
    folder_path: str = None,
):
    """
    Downloads a GOES snapshot from NASA Worldview and saves it to a file.

    :param time: The timestamp in ISO format (e.g., '2024-09-19T06:00:00Z').
    :param layer_type: The layer type, 'vis' for visible or 'inf' for infrared.
    :param extent: Extent of the snapshot.
    :param folder_path: Optional path to the folder where the file should be saved. If not provided, saves in the current directory.
    """
    if layer_type == "vis":
        layer = "GOES-East_ABI_GeoColor"
    elif layer_type == "inf":
        layer = "GOES-East_ABI_Band13_Clean_Infrared"
    else:
        raise ValueError(
            "Invalid option for layer type. Use 'vis' for visible or 'inf' for infrared."
        )

    if time is None:
        time = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    img = request_goes(time=time, layer=layer, extent=extent)

    bbox = f"{extent[2]}_{extent[0]}_{extent[3]}_{extent[1]}"
    filename = f"{time}_{bbox}_{layer}.jpg".replace(":", "")

    if folder_path:
        os.makedirs(folder_path, exist_ok=True)
        filename = os.path.join(folder_path, filename)

    img.save(filename)
    print(f"Snapshot saved as {filename}")
