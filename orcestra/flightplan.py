from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import numpy as np
import xarray as xr
import pyproj


@dataclass(frozen=True)
class LatLon:
    lat: float
    lon: float
    label: Optional[str] = None

    def towards(self, other, fraction=None, distance=None) -> LatLon:
        g = pyproj.Geod(ellps="WGS84")

        if fraction is None and distance is None:
            fraction = 0.5

        (az12, az21, dist) = g.inv(self.lon, self.lat, other.lon, other.lat)

        if distance is None:
            distance = dist * fraction

        lon, lat, _ = g.fwd(self.lon, self.lat, az12, distance)
        return LatLon(lat, lon)

    def assign_label(self, label: str) -> LatLon:
        return LatLon(self.lat, self.lon, label)


bco = LatLon(13.079773, -59.487634, "BCO")
sal = LatLon(16.73448797020352, -22.94397423993749, "SAL")


def expand_path(path: list[LatLon], dx=None, max_points=None):
    """
    NOTE: this function follows great circles
    """

    lon_points = np.asarray([p.lon for p in path])
    lat_points = np.asarray([p.lat for p in path])
    labels = [p.label for p in path]

    if len(path) < 2:
        lons = lon_points
        lats = lat_points
        dists = np.zeros_like(lon_points)
        indices = np.arange(len(lon_points))
    else:
        g = pyproj.Geod(ellps="WGS84")

        (az12, az21, dist) = g.inv(
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
        dists = []
        indices = []

        distance_so_far = 0
        indices_so_far = 0
        for lon1, lat1, lon2, lat2, n, d in zip(
            lon_points[:-1],
            lat_points[:-1],
            lon_points[1:],
            lat_points[1:],
            points_per_segment,
            dist,
        ):
            lon, lat = np.array(g.npts(lon1, lat1, lon2, lat2, n)).T
            lons.append([lon1])
            lons.append(lon)
            lats.append([lat1])
            lats.append(lat)
            dists.append(distance_so_far + np.linspace(0.0, d, n + 2)[:-1])
            indices.append(indices_so_far)
            distance_so_far += d
            indices_so_far += len(lon) + 1

        indices.append(indices_so_far - 1)
        lons = np.concatenate(lons)
        lats = np.concatenate(lats)
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
        },
    )


@dataclass
class IntoCircle:
    center: LatLon
    radius: float
    angle: float

    def __call__(self, start: LatLon, include_start: bool = False):
        g = pyproj.Geod(ellps="WGS84")

        (az12, az21, dist) = g.inv(
            start.lon, start.lat, self.center.lon, self.center.lat
        )
        angles = np.linspace(az21, az21 + self.angle, 30)
        lons, lats, rev_az = g.fwd(
            np.full_like(angles, self.center.lon),
            np.full_like(angles, self.center.lat),
            angles,
            np.full_like(angles, self.radius),
        )
        points = list(map(LatLon, lats, lons))
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


def track_len(ds):
    return float(ds.distance[-1]) - float(ds.distance[0])


def path_quickplot(path, sel_time):
    import intake
    import healpy
    import matplotlib.pylab as plt
    import matplotlib.patheffects as pe
    import cartopy.crs as ccrs
    import easygems.healpix as egh

    levels_cwv = [45, 50]
    map_extent = [-65, -5, -0, 20]

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

    ax1.coastlines(alpha=1.0)
    ax1.gridlines(
        draw_labels=True, dms=True, x_inline=False, y_inline=False, alpha=0.25
    )

    var = era5.tcwv
    egh.healpix_show(
        var.sel(time=sel_time),
        method="linear",
        alpha=0.75,
        cmap="Blues",
        vmin=45,
        vmax=70,
        ax=ax1,
    )  # for cc use vmin=0, vmax=0.1)

    contour_lines = egh.healpix_contour(
        var.sel(time=sel_time),
        levels=levels_cwv,
        colors="grey",
        linewidths=1,
        alpha=1,
        ax=ax1,
    )
    plt.clabel(contour_lines, inline=True, fontsize=8, colors="grey", fmt="%d")

    ax1.plot(
        path.lon,
        path.lat,
        transform=ccrs.PlateCarree(),
        label="HALO track",
        color="C1",
        linestyle="-",
    )

    for lon, lat, label in zip(
        path.lon[path.waypoint_indices],
        path.lat[path.waypoint_indices],
        path.waypoint_labels.values,
    ):
        ax1.annotate(
            label,
            (lon, lat),
            color="C1",
            path_effects=[pe.withStroke(linewidth=4, foreground="white")],
        )

    ax2 = fig.add_subplot(2, 1, 2)
    era_track.cc.plot(x="distance", yincrease=False, cmap="Blues", ax=ax2)
    ax2.axhline(147, color="k")

    secax = ax2.secondary_xaxis("top")
    secax.set_xlabel("waypoints")
    secax.set_xticks(pix.distance[path.waypoint_indices], path.waypoint_labels.values)
    for point in pix.distance[path.waypoint_indices]:
        ax2.axvline(point, color="k", lw=1)

    return fig
