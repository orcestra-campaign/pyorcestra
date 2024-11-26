from __future__ import annotations
import dataclasses
import pathlib
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from functools import cached_property
from typing import Optional, List, Dict
from warnings import warn
import datetime
import json
import numbers

import numpy as np
import pyproj
import scipy.signal
import xarray as xr
import pandas as pd
import topojson
from scipy.optimize import minimize
from xarray.backends import BackendEntrypoint

from .flight_performance import get_flight_performance, aircraft_performance
from .utils import parse_datestr


geod = pyproj.Geod(ellps="WGS84")


def overpass_time(
    x_track, y_track, x_lon="lon", x_lat="lat", y_lon="IRS_LON", y_lat="IRS_LAT"
):
    """
    Returns distance and time at closet overpass. Default assumes second platform is HALO
    tracks from BAHAMAS (with well formed time dimension) and has higher rate temporal data.
    """
    x = x_track.interp(time=y_track.time)
    az12, az21, dist = geod.inv(x[x_lon], x[x_lat], y_track[y_lon], y_track[y_lat])
    return dist, y_track.time[dist.argmin()]


def no_cartopy_download_warning():
    import warnings
    from cartopy.io import DownloadWarning

    warnings.filterwarnings(
        action="ignore",
        message="",
        category=DownloadWarning,
        module="cartopy",
    )


def split_deg(deg, precision):
    """
    deg: decimal degree valur
    precision: fraction of degree to round to
        e.g. 12 == 1/12th of a degree -> 60/12 == 5 minutes
    """
    base_value = np.round(deg * precision)
    return int(abs(base_value / precision)), abs(base_value * (60 / precision)) % 60


def SN(deg):
    """
    return south (S) or north (N)
    """
    return "S" if deg < 0 else "N"


def WE(deg):
    """
    return west (W) or east (E)
    """
    return "W" if deg < 0 else "E"


@dataclass(frozen=True)
class LatLon:
    lat: float
    lon: float
    label: Optional[str] = None
    fl: Optional[float] = None
    time: Optional[datetime.datetime | str] = None
    note: Optional[str] = None

    def __post_init__(self):
        if isinstance(self.time, (str, np.datetime64, xr.DataArray)):
            super().__setattr__("time", parse_datestr(self.time))
        if self.time is not None and (
            self.time.tzinfo is None or self.time.tzinfo.utcoffset(self.time) is None
        ):
            warn(
                f"Time {self.time} of {self} is naive (i.e. NOT timezone aware!). Please consider specifying a time zone to avoid confision."
            )

        if isinstance(self.lat, numbers.Real):
            super().__setattr__("lat", float(self.lat))

        if isinstance(self.lon, numbers.Real):
            super().__setattr__("lon", float(self.lon))

    def towards(self, other, fraction=None, distance=None) -> LatLon:
        if fraction is None and distance is None:
            fraction = 0.5

        (az12, az21, dist) = geod.inv(self.lon, self.lat, other.lon, other.lat)

        if distance is None:
            distance = dist * fraction

        return self.course(az12, distance)

    def course(self, direction, distance) -> LatLon:
        lon, lat, _ = geod.fwd(self.lon, self.lat, direction, distance)
        return LatLon(lat, lon)

    def assign_label(self, label: str) -> LatLon:
        return self.assign(label=label)

    def format_5min(self):
        dlat, mlat = split_deg(self.lat, 12)
        dlon, mlon = split_deg(self.lon, 12)
        return f"{dlat:02.0f}{mlat:02.0f}{SN(self.lat)}{dlon:03.0f}{mlon:02.0f}{WE(self.lon)}"

    def format_1min(self):
        dlat, mlat = split_deg(self.lat, 60)
        dlon, mlon = split_deg(self.lon, 60)
        return f"{dlat:02.0f}{mlat:02.0f}{SN(self.lat)}{dlon:03.0f}{mlon:02.0f}{WE(self.lon)}"

    def format_min100(self):
        dlat, mlat = split_deg(self.lat, 6000)
        dlon, mlon = split_deg(self.lon, 6000)
        return f"{dlat:02.0f}{mlat:05.2f}{SN(self.lat)}{dlon:03.0f}{mlon:05.2f}{WE(self.lon)}"

    def format_pilot(self):
        dlat, mlat = split_deg(self.lat, 6000)
        dlon, mlon = split_deg(self.lon, 6000)
        return f"{SN(self.lat)}{dlat:02.0f} {mlat:05.2f}, {WE(self.lon)}{dlon:03.0f} {mlon:05.2f}"

    def get_distance(self, other: LatLon) -> float:
        dist = geod.inv(self.lon, self.lat, other.lon, other.lat)[2]
        return dist

    assign = dataclasses.replace


bco = LatLon(13.16263889, -59.42875000, "BCO", fl=0)
edmo = LatLon(48.08140278, 11.2831, "EDMO", fl=0)
edja = LatLon(47.9888, 10.2395, "EDJA", fl=0)
bgi = tbpb = LatLon(13.074722, -59.4925, "TBPB", fl=0)
sal = LatLon(16.73448797020352, -22.94397423993749, "SAL", fl=0)
mindelo = LatLon(16.877810, -24.995002, "MINDELO", fl=0)


@dataclass(frozen=True)
class FlightPlan:
    path: List[LatLon | IntoCircle]
    flight_id: Optional[str] = None
    extra_waypoints: List[LatLon] = field(default_factory=list)
    crew: Dict[str, str] = field(default_factory=dict)
    aircraft: Optional[str] = None

    @cached_property
    def _simple_path_and_backmap(self):
        return simplify_path(self.path, return_backmap=True)

    @property
    def simple_path(self):
        return self._simple_path_and_backmap[0]

    @property
    def simple_backmap(self):
        return self._simple_path_and_backmap[1]

    @cached_property
    def ds(self):
        return expand_path(
            self.path, max_points=400, return_raw_points=True, aircraft=self.aircraft
        )

    @cached_property
    def circles(self):
        return [p for p in self.path if isinstance(p, IntoCircle)]

    @cached_property
    def waypoints_or_centers(self):
        return [p.center if isinstance(p, IntoCircle) else p for p in self.path]

    @cached_property
    def important_labelled_waypoints(self):
        wps = []
        for i, p in enumerate(self.path):
            if isinstance(p, IntoCircle) and p.center.label is not None:
                if not p.is_on_entry_path():
                    wps.append(self.simple_path[self.simple_backmap[i].start])
                wps.append(p.center)
            elif p.label is not None:
                wps.append(p)
        return wps

    def _require_flightlevels(self):
        for i, wp in enumerate(self.path):
            if isinstance(wp, IntoCircle):
                wp = wp.center
            if wp.fl is None:
                raise ValueError(
                    f"flight level information has to be set to proceed. It's missing in the {i}th point: {wp}"
                )

    def _require_performance(self):
        if self.aircraft is None:
            from warnings import warn

            warn(
                "The aircraft attribute of this FlightPlan is not set, but it's required for aircraft performance calculations. Currently, the code will proceed using the default aircraft performance, but the code may break in future.",
                FutureWarning,
            )
        elif self.aircraft not in aircraft_performance:
            raise ValueError(
                f"Could not find aircraft performance for aircraft '{self.aircraft}'. Currently supported aircraft are {list(sorted(aircraft_performance))}."
            )

    def _require_time(self):
        self._require_flightlevels()
        self._require_performance()
        if "time" not in self.ds:
            raise ValueError(
                "Could not compute time for this FlightPlan. You have to assign a `time` attribute to one of the waypoints along the path."
            )

    def computed_time_at_raw_index(self, i, end=False):
        self._require_time()
        if end:
            j = self.ds.raw_points_end[i]
        else:
            j = self.ds.raw_points_start[i]

        try:
            time = self.ds.time[j].values
        except AttributeError:
            raise AttributeError(
                "Could not compute time. You may be missing a waypoint with assigned reference time, or may be missing flight level information or aircraft performance information"
            )

        return (
            pd.Timestamp(time)
            .to_pydatetime(warn=False)
            .replace(tzinfo=datetime.timezone.utc)
        )

    @cached_property
    def takeoff_time(self):
        return self.computed_time_at_raw_index(0)

    @cached_property
    def landing_time(self):
        return self.computed_time_at_raw_index(-1, end=True)

    @cached_property
    def duration(self):
        self._require_flightlevels()
        self._require_performance()
        try:
            d1 = self.ds.duration.values[0]
            d2 = self.ds.duration.values[-1]
        except AttributeError:
            raise AttributeError(
                "FlightPlan.duration could not be computed. Maybe you are missing flight level or aircraft performance data?"
            )
        return pd.Timedelta(d2 - d1).to_pytimedelta()

    def show_details(self):
        print(to_detailed_txt(self))

    def export(self):
        return export_flightplan(self)

    def preview(self, **kwargs):
        return path_preview(self, **kwargs)

    def profile(self, **kwargs):
        return vertical_preview(self, **kwargs)


def attach_flight_performance(ds, performance):
    second = np.timedelta64(1000000000, "ns")
    ds = ds.assign(speed=(ds.fl.dims, performance.speed_at_fl(ds.fl)))
    segment_distance = np.diff(ds.distance.values)
    mean_speed = (ds.speed.values[:-1] + ds.speed.values[1:]) / 2
    duration = (
        np.cumsum(np.concatenate([[0], (segment_distance / mean_speed)])) * second
    )
    ds = ds.assign(duration=(ds.fl.dims, duration))
    return ds


def expand_path(
    path: list[LatLon], dx=None, max_points=None, return_raw_points=False, aircraft=None
):
    """
    NOTE: this function follows great circles
    """

    path, backmap = simplify_path(path, return_backmap=True)
    lon_points = np.asarray([p.lon for p in path])
    lat_points = np.asarray([p.lat for p in path])
    fl_points = np.asarray([p.fl if p.fl is not None else np.nan for p in path])
    labels = [p.label for p in path]

    if len(path) < 2:
        lons = lon_points
        lats = lat_points
        fls = fl_points
        dists = np.zeros_like(lon_points)
        indices = np.arange(len(lon_points))
    else:
        (az12, az21, dist) = geod.inv(
            lon_points[:-1], lat_points[:-1], lon_points[1:], lat_points[1:]
        )
        total_distance = np.sum(dist)

        n_points = None
        if dx is not None:
            n_points = total_distance / dx

        if max_points is not None:
            if n_points is None or n_points > max_points:
                n_points = max_points

        if n_points is None:
            raise ValueError("at least one of dx or max_points must be defined")

        points_per_segment = np.maximum(n_points * dist / total_distance, 1).astype(
            "int"
        )

        lons = []
        lats = []
        fls = []
        dists = []
        indices = []

        distance_so_far = 0
        indices_so_far = 0
        for lon1, lat1, fl1, lon2, lat2, fl2, n, d in zip(
            lon_points[:-1],
            lat_points[:-1],
            fl_points[:-1],
            lon_points[1:],
            lat_points[1:],
            fl_points[1:],
            points_per_segment,
            dist,
        ):
            lon, lat = np.array(geod.npts(lon1, lat1, lon2, lat2, n)).T
            lons.append([lon1])
            lons.append(lon)
            lats.append([lat1])
            lats.append(lat)
            fls.append(np.linspace(fl1, fl2, n + 2)[:-1])
            dists.append(distance_so_far + np.linspace(0.0, d, n + 2)[:-1])
            indices.append(indices_so_far)
            distance_so_far += d
            indices_so_far += len(lon) + 1

        lons.append([lon_points[-1]])
        lats.append([lat_points[-1]])
        fls.append([fl_points[-1]])
        dists.append([distance_so_far])
        indices.append(indices_so_far)

        lons = np.concatenate(lons)
        lats = np.concatenate(lats)
        fls = np.concatenate(fls)
        dists = np.concatenate(dists)

        simple_path_indices = np.array(indices)
        il = list(
            zip(*[(i, label) for i, label in zip(indices, labels) if label is not None])
        )
        if len(il) > 0:
            indices, labels = il
        else:
            indices = []
            labels = []
        indices = np.array(indices)
        labels = np.array(labels)

    ds = xr.Dataset(
        {
            "waypoint_indices": ("waypoint", indices),
            "waypoint_labels": ("waypoint", labels),
        },
        coords={
            "distance": ("distance", dists),
            "lon": ("distance", lons),
            "lat": ("distance", lats),
            "fl": ("distance", fls),
        },
    )

    if return_raw_points:
        ds = ds.assign(
            {
                "raw_points_start": (
                    "raw_points",
                    simple_path_indices[
                        [i.start if isinstance(i, slice) else i for i in backmap]
                    ],
                ),
                "raw_points_end": (
                    "raw_points",
                    simple_path_indices[
                        [i.stop - 1 if isinstance(i, slice) else i for i in backmap]
                    ],
                ),
            }
        )

    if performance := get_flight_performance(aircraft=aircraft):
        ds = ds.pipe(attach_flight_performance, performance)

    points_with_time = [(i, p) for i, p in enumerate(path) if p.time is not None]
    if len(points_with_time) > 1:
        raise ValueError(
            "Multiple waypoints have an associated time. Currently, only a single point with an associated time is implemented!"
        )
    elif len(points_with_time) == 1:
        i, point = points_with_time[0]
        if point.time.tzinfo is None or point.time.tzinfo.utcoffset(point.time) is None:
            warn(
                f"Time {point.time} of {point} is naive (i.e. NOT timezone aware!). Assuming UTC."
            )
            reftime = np.datetime64(point.time)
        else:
            reftime = np.datetime64(
                point.time.astimezone(datetime.timezone.utc).replace(tzinfo=None)
            )

        if "duration" in ds:
            offset = ds.duration.values[simple_path_indices[i]]
            ds = ds.assign(time=ds.duration - offset + reftime)

    return ds


def _az_towards_point_with_angle(A, B, alpha, radius):
    r"""
    Find azimuth value from B such that the angle from B to a A
    around an unknown point X is alpha, such and the distance
    between B and X is radius.

    A ------------ X
                ^   \
           angle \__ \   <-radius
                      \
                       B
    """

    def cost(angle):
        lon, lat, rev_az = geod.fwd(B.lon, B.lat, angle, radius)
        (azA, azB), _, _ = geod.inv(
            [lon] * 2, [lat] * 2, [A.lon, B.lon], [A.lat, B.lat]
        )
        return (alpha - ((azA - azB + 180) % 360 - 180)) ** 2

    az12, _, _ = geod.inv(A.lon, A.lat, B.lon, B.lat)
    res = minimize(cost, az12 + 180 - alpha)

    if not res.success:
        raise ValueError(
            f"could not find point X for angle {alpha}, reason: {res.message}"
        )
    return res.x[0]


@dataclass
class IntoCircle:
    center: LatLon
    radius: float
    angle: float
    enter: Optional[float] = None

    def __post_init__(self):
        assert (
            self.center.time is None
        ), "The time attribute of the center coordinate of a circle MUST be None. I.e. a circle will have a duration, thus you can't assign it a point in time."

    def is_on_entry_path(self):
        """
        Returns true, if circle center is on the flight path into the circle.
        """
        return self.enter is None or self.enter % 180 == 90

    def is_entry_through(self):
        """
        Returns true, if entry into circle first crosses the circle.
        """
        if self.enter is None:
            return False
        return (self.enter + 180) % 360 - 180 > 0

    def get_intersect(self, other: IntoCircle):
        fwd, _, distance = geod.inv(
            self.center.lon, self.center.lat, other.center.lon, other.center.lat
        )
        if distance > self.radius + other.radius:
            raise ValueError(
                "The two circles are too far apart. There is no intersection"
            )
        else:

            def cost(fwd_az):
                fwd_az = (fwd_az + 180) % 360 - 180
                cand_lon, cand_lat, _ = geod.fwd(
                    self.center.lon, self.center.lat, fwd_az, self.radius
                )
                _, _, dist = geod.inv(
                    cand_lon, cand_lat, other.center.lon, other.center.lat
                )
                return (dist - other.radius) ** 2

            def min_angle(x0):
                res = minimize(cost, x0, bounds=[(-180, 180)])
                if not res.success:
                    raise ValueError("could not find intersection point")
                return res

            def get_latlon(res):
                lon, lat, _ = geod.fwd(
                    self.center.lon,
                    self.center.lat,
                    (res.x + 180) % 360 - 180,
                    self.radius,
                )
                return LatLon(lon=lon, lat=lat)

        res1 = min_angle(fwd)
        res2 = min_angle(-res1.x)
        return (get_latlon(res1), get_latlon(res2))

    def __call__(self, start: LatLon, include_start: bool = False):
        if self.enter is None:
            (_, start_angle, _) = geod.inv(
                start.lon, start.lat, self.center.lon, self.center.lat
            )
        else:
            try:
                start_angle = _az_towards_point_with_angle(
                    start, self.center, 90 - self.enter, self.radius
                )
            except ValueError:
                raise ValueError(
                    f"could not find a solution for circle {self} with start point {start}, maybe start is inside cirle. You may want to try without specifying `enter`."
                )

        angles = np.linspace(start_angle, start_angle + self.angle, 90)
        lons, lats, rev_az = geod.fwd(
            np.full_like(angles, self.center.lon),
            np.full_like(angles, self.center.lat),
            angles,
            np.full_like(angles, self.radius),
        )
        points = [LatLon(lat, lon, fl=self.center.fl) for lat, lon in zip(lats, lons)]
        if self.center.label is not None:
            points[0] = points[0].assign_label(f"{self.center.label}_in")
            points[-1] = points[-1].assign_label(f"{self.center.label}_out")
        if include_start:
            points = [start] + points
        return points


def simplify_path(path, return_backmap=False):
    backmap = []

    def _gen():
        last = None
        i = 0
        for p in path:
            if callable(p):
                first = i
                for last in p(last):
                    yield last
                    i += 1
                backmap.append(slice(first, i))
            else:
                last = p
                yield last
                backmap.append(i)
                i += 1

    out_path = list(_gen())
    if return_backmap:
        return out_path, backmap
    else:
        return out_path


def path_as_ds(path):
    if isinstance(path, FlightPlan):
        return path.ds
    elif isinstance(path, list):
        return FlightPlan(path).ds
    else:
        return path


def path_len(path):
    ds = path_as_ds(path)
    return float(ds.distance[-1]) - float(ds.distance[0])


def track_len(ds):
    warn(
        "track_len is deprecated, please use path_len instead",
        DeprecationWarning,
        stacklevel=2,
    )
    return path_len(ds)


def first_point(path):
    ds = path_as_ds(path)
    lat = float(ds.lat.values[0])
    lon = float(ds.lon.values[0])
    return LatLon(lat, lon)


def last_point(path):
    ds = path_as_ds(path)
    lat = float(ds.lat.values[-1])
    lon = float(ds.lon.values[-1])
    return LatLon(lat, lon)


def plot_path(plan, ax, color="C1", label=None, show_waypoints=True, extra_color="C3"):
    import cartopy.crs as ccrs

    ds = path_as_ds(plan)

    ax.plot(ds.lon, ds.lat, transform=ccrs.Geodetic(), label=label, color=color)

    if show_waypoints:
        import matplotlib.patheffects as pe

        import textalloc as ta
        from matplotlib.colors import to_rgba, to_hex

        if isinstance(plan, FlightPlan):
            labels = set(
                [(p.lon, p.lat, p.label) for p in plan.important_labelled_waypoints]
            )

            extra_points = set([(p.lon, p.lat, p.label) for p in plan.extra_waypoints])
        else:
            labels = set(
                zip(
                    ds.lon[ds.waypoint_indices].values,
                    ds.lat[ds.waypoint_indices].values,
                    ds.waypoint_labels.values,
                )
            )
            extra_points = set()

        if extra_points:
            elon, elat, _ = zip(*extra_points)
            ax.scatter(
                elon,
                elat,
                transform=ccrs.Geodetic(),
                color=extra_color,
                marker="+",
                s=10,
            )

        # deduplicate labels
        lon, lat, text = zip(*labels, *extra_points)

        label_color = to_rgba(color)
        line_color = label_color[:3] + (label_color[3] * 0.5,)
        extra_label_color = to_rgba(extra_color)
        extra_line_color = extra_label_color[:3] + (extra_label_color[3] * 0.5,)

        ta.allocate(
            ax,
            lon,
            lat,
            text,
            x_lines=[ds.lon],
            y_lines=[ds.lat],
            linecolor=[
                *([to_hex(line_color, True)] * len(labels)),
                *([to_hex(extra_line_color, True)] * len(extra_points)),
            ],
            textcolor=[
                *([to_hex(label_color, True)] * len(labels)),
                *([to_hex(extra_label_color, True)] * len(extra_points)),
            ],
            path_effects=[pe.withStroke(linewidth=4, foreground="white")],
        )


def plot_usurf(var, ax=None, levels=None):
    import matplotlib.pylab as plt
    import easygems.healpix as egh

    levels = levels or [0, 3]
    egh.healpix_show(
        var,
        method="linear",
        alpha=0.75,
        cmap="YlGn",
        vmin=0,
        vmax=15,
        ax=ax,
    )

    contour_lines = egh.healpix_contour(
        var,
        levels=levels,
        colors="red",
        linewidths=1,
        alpha=1,
        ax=ax,
    )
    plt.clabel(contour_lines, inline=True, fontsize=8, colors="red", fmt="%d")


def plot_cwv(var, ax=None, levels=None):
    import matplotlib.pylab as plt
    import easygems.healpix as egh

    levels = levels or [45, 50]
    egh.healpix_show(
        var,
        method="linear",
        alpha=0.75,
        cmap="Blues",
        vmin=45,
        vmax=70,
        ax=ax,
    )

    contour_lines = egh.healpix_contour(
        var,
        levels=levels,
        colors="grey",
        linewidths=1,
        alpha=1,
        ax=ax,
    )
    plt.clabel(contour_lines, inline=True, fontsize=8, colors="grey", fmt="%d")


def plot_fir(ax=None, color="#F08080", linestyle=(0, (2, 6)), linewidth=0.8, **kwargs):
    import cartopy.crs as ccrs
    import matplotlib.pylab as plt

    if ax is None:
        ax = plt.gca()

    with open(pathlib.Path(__file__).parent / "data" / "worldfirs.json", "r") as fp:
        data = json.load(fp)

    topo = topojson.Topology(data, object_name="data")
    firs = topo.to_gdf()

    lines = []
    for i, row in firs.iterrows():
        for geom in getattr(row.geometry.boundary, "geoms", [row.geometry.boundary]):
            lon, lat = zip(*list(geom.coords))
            line = ax.plot(
                lon,
                lat,
                transform=ccrs.PlateCarree(),
                color=color,
                linestyle=linestyle,
                linewidth=linewidth,
                **kwargs,
            )
            lines.append(line)

    return lines


def vertical_preview(path, color="C1", lw=2, **kwargs):
    import matplotlib.pylab as plt
    import pandas as pd
    import seaborn as sns

    path = path_as_ds(path)

    fig, ax = plt.subplots(figsize=(15, 6))
    sns.despine(offset=10)

    secax = ax.secondary_xaxis("top")
    secax.set_xlabel("waypoints")

    if "time" in path:
        x = path.time
        wpx = x[path.waypoint_indices]
        ax.set_xticks(
            wpx, labels=[f"{x:%H:%M}" for x in pd.to_datetime(wpx)], rotation=60.0
        )
        ax.set_xlabel("Time / UTC")
    else:
        x = path.distance
        wpx = x[path.waypoint_indices]
        ax.set_xticks(
            path.distance[path.waypoint_indices],
            labels=[f"{x/1000:.0f}" for x in wpx],
            rotation=60.0,
        )
        ax.set_xlabel("Distance / km")

    secax.set_xticks(
        wpx,
        path.waypoint_labels.values,
        rotation="vertical",
    )

    ax.set_yticks([200, 350, 410, 450])
    ax.set_ylabel("Flight Level / hft")

    ax.plot(x, path.fl, color=color, lw=lw, **kwargs)


def path_preview(
    plan, coastlines=True, gridlines=True, ax=None, size=8, aspect=16 / 9, **kwargs
):
    import matplotlib.pylab as plt
    import cartopy.crs as ccrs

    path = path_as_ds(plan)

    lon_min = path.lon.values.min()
    lon_max = path.lon.values.max()
    lat_min = path.lat.values.min()
    lat_max = path.lat.values.max()

    dlon = lon_max - lon_min
    dlat = lat_max - lat_min

    if dlon / dlat > aspect:
        dlat = dlon / aspect
    else:
        dlon = dlat * aspect

    clon = (lon_min + lon_max) / 2
    clat = (lat_min + lat_max) / 2

    map_extent = [
        clon - 0.6 * dlon,
        clon + 0.6 * dlon,
        clat - 0.6 * dlat,
        clat + 0.6 * dlat,
    ]

    if ax is None:
        fig, ax = plt.subplots(
            figsize=(size * aspect, size),
            subplot_kw={"projection": ccrs.PlateCarree()},
        )
    ax.set_extent(map_extent, crs=ccrs.PlateCarree())

    if coastlines:
        no_cartopy_download_warning()
        ax.coastlines(alpha=1.0)
    if gridlines:
        no_cartopy_download_warning()
        ax.gridlines(draw_labels=True, alpha=0.25)

    plot_path(plan, ax=ax, label="path", **kwargs)
    return ax


def path_quickplot(path, sel_time, crossection=True):
    import intake
    import healpix
    import matplotlib.pylab as plt
    import cartopy.crs as ccrs

    map_extent = [-65, -5, -0, 20]

    path = path_as_ds(path)

    cat = intake.open_catalog("https://tcodata.mpimet.mpg.de/internal.yaml")
    era5 = cat.HERA5(time="PT1H").to_dask()

    pix = xr.DataArray(
        healpix.ang2pix(2**7, path.lon, path.lat, nest=True, lonlat=True),
        dims=("distance",),
        name="pix",
    )

    era_track = era5.sel(time=sel_time).isel(cell=pix)

    fig = plt.figure(figsize=(15, 12))
    ax1 = fig.add_subplot(2, 1, 1, projection=ccrs.PlateCarree())
    ax1.set_extent(map_extent, crs=ccrs.PlateCarree())

    no_cartopy_download_warning()
    ax1.coastlines(alpha=1.0)
    ax1.gridlines(
        draw_labels=True, dms=True, x_inline=False, y_inline=False, alpha=0.25
    )

    plot_cwv(era5.tcwv.sel(time=sel_time), ax=ax1)

    plot_path(path, ax=ax1, label="HALO track", color="C1")

    if crossection:
        ax2 = fig.add_subplot(2, 1, 2)
        era_track.cc.plot(x="distance", yincrease=False, cmap="Blues", ax=ax2)
        ax2.axhline(147, color="k")

        secax = ax2.secondary_xaxis("top")
        secax.set_xlabel("waypoints")
        secax.set_xticks(
            pix.distance[path.waypoint_indices], path.waypoint_labels.values
        )
        for point in pix.distance[path.waypoint_indices]:
            ax2.axvline(point, color="k", lw=1)

    return fig


def to_kml(path):
    import simplekml

    ds = path_as_ds(path)

    # fl2m = 100 * .3048  # currently not used: can lead to parallax error
    kml = simplekml.Kml()
    kml.newlinestring(
        name="Flight Track",
        description="Flight Track",
        coords=list(zip(ds.lon.values, ds.lat.values, np.zeros_like(ds.fl.values))),
    )
    seen_points = set()
    for ip, it in enumerate(ds.waypoint_indices.values):
        p = name, lon, lat = (
            ds.waypoint_labels.values[ip],
            ds.lon.values[it],
            ds.lat.values[it],
        )
        if p not in seen_points:
            kml.newpoint(name=name, coords=[(lon, lat)])
            seen_points.add(p)
    return kml.kml()


def to_gpx(path):
    import gpxpy

    gpx = gpxpy.gpx.GPX()

    gpx_track = gpxpy.gpx.GPXTrack()
    gpx.tracks.append(gpx_track)

    gpx_segment = gpxpy.gpx.GPXTrackSegment()
    gpx_track.segments.append(gpx_segment)

    ds = path_as_ds(path)
    for lat, lon, fl in zip(ds.lat.values, ds.lon.values, ds.fl.values):
        gpx_segment.points.append(
            gpxpy.gpx.GPXTrackPoint(lat, lon, elevation=fl * 100 * 0.3048)
        )

    return gpx.to_xml()


def to_geojson(path):
    import json

    ds = path_as_ds(path)

    return json.dumps(
        {
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "geometry": {
                        "type": "LineString",
                        "coordinates": [
                            [float(lon), float(lat)]
                            for lon, lat in zip(ds.lon.values, ds.lat.values)
                        ],
                    },
                    "properties": {},
                },
                *[
                    {
                        "type": "Feature",
                        "geometry": {
                            "type": "Point",
                            "coordinates": [
                                float(ds.lon.values[it]),
                                float(ds.lat.values[it]),
                            ],
                        },
                        "properties": {"label": str(ds.waypoint_labels.values[ip])},
                    }
                    for ip, it in enumerate(ds.waypoint_indices.values)
                ],
            ],
        }
    )


def az_to_text(az):
    sections = [
        "north",
        "north east",
        "east",
        "south east",
        "south",
        "south west",
        "west",
        "north west",
    ]
    return sections[int(((az + (360 / 2 / len(sections))) * len(sections) / 360) % 8)]


def to_detailed_txt(plan: FlightPlan):
    from io import StringIO

    file = StringIO()

    file.write("Detailed Overview:\n")
    for i, point in enumerate(plan.path):
        if isinstance(point, LatLon):
            if i == 0:
                prefix = ""
            else:
                prefix = "to"
            file.write(
                f"{prefix:13s} {point.label:12s} {point.format_pilot():20s}, FL{point.fl:03d}, {plan.computed_time_at_raw_index(i):%H:%M:%S %Z}, {point.note or ''}\n"
            )
        elif isinstance(point, IntoCircle):
            tstart = plan.computed_time_at_raw_index(i)
            tend = plan.computed_time_at_raw_index(i, end=True)
            radius_nm = point.radius / 1852
            direction = "CW" if point.angle > 0 else "CCW"
            simple_slice = plan.simple_backmap[i]
            first_point = plan.simple_path[simple_slice.start]
            (az12, az21, dist) = geod.inv(
                first_point.lon,
                first_point.lat,
                point.center.lon,
                point.center.lat,
            )
            if not point.is_on_entry_path():
                file.write(
                    f"{'to (helper)':13s} {first_point.label:12s} {first_point.format_pilot():20s}, FL{first_point.fl:03d}, {tstart:%H:%M:%S %Z}, upcoming circle will be entered here\n"
                )
            notes = [
                f"radius: {radius_nm:.0f} nm",
                f"{abs(point.angle):.0f}° {direction}",
                f"enter from {az_to_text(az21)}",
            ]
            if point.is_entry_through():
                notes.append("fly through circle before entering")
            if point.center.note:
                notes.append(point.center.note)
            file.write(
                f"circle around {point.center.label:12s} {point.center.format_pilot():20s}, FL{point.center.fl:03d}, {tstart:%H:%M:%S %Z} - {tend:%H:%M:%S %Z}, {', '.join(notes)}\n"
            )

    return file.getvalue()


def to_txt(plan: FlightPlan):
    from io import StringIO

    file = StringIO()

    file.write(f"Flight {plan.flight_id}\n\n")
    #
    # DM Format
    file.write("------------------------------------------------------------\n")
    file.write("\nDM Format:\n")
    file.write(
        " ".join(point.format_1min() for point in plan.waypoints_or_centers) + "\n"
    )
    for point in plan.extra_waypoints:
        file.write(f"Extra waypoint: {point.format_1min()}\n")
    #
    # DM.mm format
    file.write("\n------------------------------------------------------------\n")
    file.write("\nDMmm Format:\n")
    for point in plan.waypoints_or_centers:
        file.write(f"{point.format_pilot()}, {point.label}\n")
    file.write("\n-- extra waypoints:\n")
    for point in plan.extra_waypoints:
        file.write(f"{point.format_pilot()}, {point.note or ''}\n")

    #
    # Detailed overview with notes
    file.write("\n------------------------------------------------------------\n\n\n")
    file.write(to_detailed_txt(plan))

    file.write("\n\nCrew:")
    for position, person in plan.crew.items():
        file.write(f"\n{position:22s} {person}")

    return file.getvalue()


def as_href(data, mime):
    import base64

    return f"data:{mime};base64,{base64.b64encode(data).decode('ascii')}"


def export_flightplan(flight_id_or_plan, plan=None):
    if not plan:
        plan = flight_id_or_plan
        flight_id = plan.flight_id
    else:
        flight_id = flight_id_or_plan

    if flight_id is None:
        raise ValueError("You have to set a flight_id to export the flight plan.")

    from ipywidgets import HTML
    from IPython.display import display

    exports = [
        (
            ".geojson",
            "GeoJSON",
            "application/geo+json",
            to_geojson(plan).encode("utf-8"),
        ),
        (
            ".kml",
            "KML",
            "application/vnd.google-earth.kml+xml",
            to_kml(plan).encode("utf-8"),
        ),
        (
            ".gpx",
            "GPX",
            "application/gpx+xml",
            to_gpx(plan).encode("utf-8"),
        ),
    ]
    if isinstance(plan, FlightPlan) and "time" in plan.ds:
        exports.append(
            ("_waypoints.txt", "TXT", "text/plain", to_txt(plan).encode("utf-8"))
        )

    # BUTTONS
    html = """<html>
    <head>
    <meta name="viewport" content="width=device-width, initial-scale=1">
    </head>
    <body>
    <h2>
    Download Flightplan as:
    </h2>
    """
    for suffix, name, mime, data in exports:
        html += f"""
            <a download="{flight_id}{suffix}" href="{as_href(data, mime)}" download>
            <button class="p-Widget jupyter-widgets jupyter-button widget-button mod-warning">{name}</button>
            </a>
            """
    html += """
    </body>
    </html>
    """
    return display(HTML(html))


def open_ftml(path):
    """Return an MSS flight track in FTML format as dataset."""
    tree = ET.parse(path)
    root = tree.getroot()
    waypoints = root[0]

    data = {}
    for attr in ["lat", "lon", "flightlevel", "location"]:
        data[attr] = [
            float(way.attrib[attr]) if attr != "location" else way.attrib[attr]
            for way in waypoints
        ]

    ds = xr.Dataset(
        data_vars={k: ("waypoint", v) for k, v in data.items()},
        coords={"waypoint": np.arange(len(waypoints))},
    )

    return ds


class FlightTrackEntrypoint(BackendEntrypoint):
    def open_dataset(self, filename_or_obj, *, drop_variables=None):
        return open_ftml(filename_or_obj)

    open_dataset_parameters = ["filename_or_obj"]

    def guess_can_open(self, filename_or_obj):
        try:
            ext = pathlib.Path(filename_or_obj).suffix
        except TypeError:
            return False
        return ext in {".ftml"}

    description = "Use .ftml files in Xarray"

    url = "https://github.com/orcestra-campaign/pyorcestra"


def calc_zonal_mean(field, lon_min, lon_max, lat_min, lat_max):
    import easygems.healpix as egh

    bbox = [lon_min, lon_max, lat_min, lat_max]

    in_bbox = egh.isel_extent(field, bbox)

    field_bbox = field.sel(
        cell=in_bbox,
    )

    field_lat = field_bbox.groupby(field_bbox.lat).mean()
    return field_lat


# Function that finds longitude of ec track that corresponds to the provided latitude
def find_ec_lon(lat_sel, ec_lons, ec_lats):
    if not np.all(np.diff(ec_lats) > 0):
        ec_lons = ec_lons[::-1]
        ec_lats = ec_lats[::-1]
    assert np.all(np.diff(ec_lats) > 0), "ec_lats are not monotonic"
    return np.interp(lat_sel, ec_lats, ec_lons)


def point_on_track(track, lat=None, lon=None, with_time=False):
    assert (lat is None and lon is not None) or (lat is not None and lon is None)

    lats = track.lat.values
    lons = track.lon.values

    if lat is not None:
        if not np.all(np.diff(lats) > 0):
            lats = lats[::-1]
            lons = lons[::-1]
        assert np.all(np.diff(lats) > 0), "latitude values are not monotonic"
        assert lats[0] <= lat <= lats[-1], "requested latitude is not in range"
        point = LatLon(lat, float(np.interp(lat, lats, lons)))
    elif lon is not None:
        if not np.all(np.diff(lons) > 0):
            lats = lats[::-1]
            lons = lons[::-1]
        assert np.all(np.diff(lons) > 0), "longitude values are not monotonic"
        assert lons[0] <= lon <= lons[-1], "requested longitude is not in range"
        point = LatLon(float(np.interp(lon, lons, lats)), lon)

    if with_time:
        e = np.datetime64("2024-08-01")
        s = np.timedelta64(1, "ns")
        if lat is not None:
            time = ((track.swap_dims({"time": "lat"}).time - e) / s).interp(
                lat=lat
            ) * s + e
        elif lon is not None:
            time = ((track.swap_dims({"time": "lon"}).time - e) / s).interp(
                lon=lon
            ) * s + e
        point = point.assign(time=time.values)

    return point


def ec_time_at_lat(ec_track, lat):
    e = np.datetime64("2024-08-01")
    s = np.timedelta64(1, "ns")
    return ((ec_track.swap_dims({"time": "lat"}).time - e) / s).interp(lat=lat) * s + e


def find_edges(cwv, cwv_thresh, cwv_min=0, lat_cwv_max=9.0):
    """
    Determine latitude of peak in CWV that is closest to the latitude of peak CWV in the average CWV profile (lat_cwv_max).
    Assess where the moist tropics end by dropping all latitudes where CWV drops below cwv_min.
    Within the remaining moist band, assess the northernmost and southernmost latitude at which CWV is equal to cwv_thresh.

    If CWV is below cwv_thresh everywhere, return NAN values.
    """

    if cwv.max().values <= cwv_thresh:
        lat_north, lat_south = np.nan, np.nan

    else:
        peaks_i, peaks_props = scipy.signal.find_peaks(
            cwv, height=cwv_thresh, prominence=2
        )

        if len(peaks_i) == 0:
            lat_north, lat_south = np.nan, np.nan

        else:
            dist_peaks = np.abs(lat_cwv_max - cwv.lat[peaks_i])
            cwv_lat_max = dist_peaks.lat[np.argmin(dist_peaks.values)]

            cwv_north = cwv.where(
                (cwv.lat >= cwv_lat_max).compute() & (cwv > cwv_min).compute(),
                drop=True,
            )
            cwv_south = cwv.where(
                (cwv.lat <= cwv_lat_max).compute() & (cwv > cwv_min).compute(),
                drop=True,
            )

            lat_north = float(cwv_north.lat.where(cwv_north <= cwv_thresh).min().values)
            lat_south = float(cwv_south.lat.where(cwv_south >= cwv_thresh).min().values)

    return lat_south, lat_north
