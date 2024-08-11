from __future__ import annotations
import dataclasses
import pathlib
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import Optional
from warnings import warn

import numpy as np
import pyproj
import scipy.signal
import xarray as xr
from scipy.optimize import minimize
from xarray.backends import BackendEntrypoint


geod = pyproj.Geod(ellps="WGS84")


def no_cartopy_download_warning():
    import warnings
    from cartopy.io import DownloadWarning

    warnings.filterwarnings(
        action="ignore",
        message="",
        category=DownloadWarning,
        module="cartopy",
    )


@dataclass(frozen=True)
class LatLon:
    lat: float
    lon: float
    label: Optional[str] = None
    fl: Optional[float] = None

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

    assign = dataclasses.replace


bco = LatLon(13.079773, -59.487634, "BCO", fl=0)
sal = LatLon(16.73448797020352, -22.94397423993749, "SAL", fl=0)


def expand_path(path: list[LatLon], dx=None, max_points=None):
    """
    NOTE: this function follows great circles
    """

    path = simplify_path(path)
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

    return xr.Dataset(
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

        angles = np.linspace(start_angle, start_angle + self.angle, 30)
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


def simplify_path(path):
    def _gen():
        last = None
        for p in path:
            if callable(p):
                for last in p(last):
                    yield last
            else:
                last = p
                yield last

    return list(_gen())


def path_as_ds(path):
    if isinstance(path, list):
        return expand_path(path, max_points=400)
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


def plot_path(path, ax, color=None, label=None, show_waypoints=True):
    import cartopy.crs as ccrs

    path = path_as_ds(path)

    ax.plot(path.lon, path.lat, transform=ccrs.Geodetic(), label=label, color=color)

    if show_waypoints:
        import matplotlib.patheffects as pe

        import textalloc as ta
        from matplotlib.colors import to_rgba, to_hex

        # deduplicate labels
        lon, lat, text = zip(
            *set(
                zip(
                    path.lon[path.waypoint_indices].values,
                    path.lat[path.waypoint_indices].values,
                    path.waypoint_labels.values,
                )
            )
        )

        label_color = to_rgba(color)
        line_color = label_color[:3] + (label_color[3] * 0.5,)

        ta.allocate(
            ax,
            lon,
            lat,
            text,
            x_lines=[path.lon],
            y_lines=[path.lat],
            linecolor=to_hex(line_color, True),
            textcolor=to_hex(label_color, True),
            path_effects=[pe.withStroke(linewidth=4, foreground="white")],
        )


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


def vertical_preview(path):
    import matplotlib.pylab as plt

    path = path_as_ds(path)

    fig, ax = plt.subplots(figsize=(15, 6))

    secax = ax.secondary_xaxis("top")
    secax.set_xlabel("waypoints")
    secax.set_xticks(
        path.distance[path.waypoint_indices],
        path.waypoint_labels.values,
        rotation="vertical",
    )
    for point in path.distance[path.waypoint_indices]:
        ax.axvline(point, color="k", lw=1)
    ax.plot(path.distance, path.fl, color="C1", lw=2)


def path_preview(path, coastlines=True, gridlines=True, ax=None, size=8, aspect=16 / 9):
    import matplotlib.pylab as plt
    import cartopy.crs as ccrs

    path = path_as_ds(path)

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

    plot_path(path, ax=ax, label="path", color="C1")
    return ax


def path_quickplot(path, sel_time, crossection=True):
    import intake
    import healpy
    import matplotlib.pylab as plt
    import cartopy.crs as ccrs

    map_extent = [-65, -5, -0, 20]

    path = path_as_ds(path)

    cat = intake.open_catalog("https://tcodata.mpimet.mpg.de/internal.yaml")
    era5 = cat.HERA5(time="PT1H").to_dask()

    pix = xr.DataArray(
        healpy.ang2pix(2**7, path.lon, path.lat, nest=True, lonlat=True),
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
                (cwv.lat >= cwv_lat_max) & (cwv > cwv_min), drop=True
            )
            cwv_south = cwv.where(
                (cwv.lat <= cwv_lat_max) & (cwv > cwv_min), drop=True
            )

            lat_north = float(
                cwv_north.lat.where(cwv_north <= cwv_thresh).min().values
            )
            lat_south = float(
                cwv_south.lat.where(cwv_south >= cwv_thresh).min().values
            )

    return lat_south, lat_north
