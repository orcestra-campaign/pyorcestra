import numpy as np
import xarray as xr
import json
import os
import pandas as pd

# Get the directory of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Construct the path to config.json relative to the script's location
config_path = os.path.join(script_dir, "config.json")

# read config.json file for parameters
with open(config_path) as f:
    config = json.load(f)


def _noise_filter_radar(ds):
    """Filter radar data by noise level.

    Parameters
    ----------
    ds : xr.Dataset
        Level0 radar dataset.

    Returns
    -------
    xr.Dataset
        Radar data filtered by noise level.

    """
    return ds.where(ds.npw1 > config["noise_threshold"])


def _state_filter_radar(ds):
    """Filter radar data for valid state.

    Parameters
    ----------
    ds : xr.Dataset
        Level0 radar dataset.

    Returns
    -------
    xr.Dataset
        Radar data filtered for valid state.
    """

    return ds.where(ds.grst == config["valid_radar_state"])


def _roll_filter(ds, roll):
    """Filter any dataset by plane roll angle.

    Parameters
    ----------
    ds : xr.Dataset
        Level0 dataset.
    roll: xr.DataArray
        Flight roll angle in degrees.

    Returns
    -------
    xr.Dataset
        Dataset filtered by plane roll angle.
    """

    roll_subsampled = roll.sel(time=ds.time, method="nearest").assign_coords(
        time=ds.time
    )

    return ds.where(np.abs(roll_subsampled) < config["roll_threshold"])


def _altitude_filter(ds, height):
    """Filter any dataset by plane altitude.

    Parameters
    ----------
    ds : xr.Dataset
        Level0 dataset.
    height: xr.DataArray
        Flight altitude in m.

    Returns
    -------
    xr.Dataset
        Dataset filtered by plane altitude.
    """

    height_subsampled = height.sel(time=ds.time, method="nearest").assign_coords(
        time=ds.time
    )

    return ds.where(height_subsampled > config["altitude_threshold"])


def _trim_dataset(ds, dim="time"):
    """
    Trim the dataset by removing data at the beginning and end until the first and last occurrence of valid data.

    Parameters
    ----------
    ds : xr.Dataset
        The input dataset.
    dim : str
        The dimension along which to trim the dataset. Default is "time".

    Returns
    -------
    xr.Dataset
        The trimmed dataset.
    """
    # Drop NaNs along the specified dimension
    valid_data = ds.dropna(dim=dim, how="all")

    # Find the first and last indices of valid data
    first_valid_index = valid_data[dim].values[0]
    last_valid_index = valid_data[dim].values[-1]

    # Slice the dataset to include only the valid data
    trimmed_ds = ds.sel({dim: slice(first_valid_index, last_valid_index)})

    return trimmed_ds


def _filter_spikes(ds, threshold=5, window=1200):
    """
    Filters out spikes in a time series by comparing the difference between each point
    and the minimum of the surrounding 5 minutes.

    Parameters
    ----------
    ds : xr.DataArray
        DataArray to filter.
    threshold : float
        Maximum allowed difference between the data and the minimum within the window.
    window : int
        Size of the window in seconds to compare the data, default is 5 minutes.

    Returns
    -------
    xr.DataArray
        Filtered DataArray.
    """
    diff = ds - ds.rolling(time=window, center=True, min_periods=1).min()
    filtered = ds.where(abs(diff) < threshold)
    interpolated = filtered.interpolate_na("time", method="linear")
    return xr.where(abs(diff) < threshold, ds, interpolated)


def _filter_land(ds, sea_land_mask, lat, lon, offset=pd.Timedelta("7s")):
    """Filters out data that was collected over land.

    Removes data by offset earlier than the the time the plane flies over land
    to avoid including land measurements due to tilt of the plane or mask inaccuracies.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset to filter.
    sea_land_mask : xr.DataArray
        Mask of land and sea. 1 for sea, 0 for land.
    lat : xr.DataArray
        Latitudes of the path.
    lon : xr.DataArray
        Longitudes of the path.
    offset : pd.Timedelta
        Time offset to remove data before the plane flies over land. Default is 7 seconds.

    Returns
    -------
    xr.Dataset
        Filtered dataset.
    """

    mask_path = sea_land_mask.sel(lat=lat, lon=lon, method="nearest")
    mask_path_subsampled = mask_path.sel(time=ds.time, method="nearest").assign_coords(
        time=ds.time
    )
    diff = (mask_path_subsampled * 1).diff("time")
    start_land = diff.where(diff == -1).dropna("time").time
    end_land = diff.where(diff == 1).dropna("time").time
    for t in start_land:
        mask_path_subsampled.loc[dict(time=slice(t - offset, t))] = 0
    for t in end_land:
        mask_path_subsampled.loc[dict(time=slice(t, t + offset))] = 0
    return ds.where(mask_path_subsampled == 1)


def filter_radar(ds, roll):
    """Filter radar data for noise, valid radar states, and roll angle.

    Parameters
    ----------
    ds : xr.Dataset
        Level0 radar dataset.

    Returns
    -------
    xr.Dataset
        Radar data filtered for noise, state, and roll angle.
    """

    return (
        ds.pipe(_noise_filter_radar)
        .pipe(_state_filter_radar)
        .pipe(_roll_filter, roll)
        .pipe(_trim_dataset)
    )


def filter_radiometer(ds, height, roll, lat, lon, mask):
    """Filter radiometer data for height and roll angle.

    Parameters
    ----------
    ds : xr.Dataset
        Level0 radiometer dataset.
    height : xr.DataArray
        Flight altitude in m.
    roll : xr.DataArray
        Flight roll angle in degrees.
    lat : xr.DataArray
        Latitudes of the flightpath.
    lon : xr.DataArray
        Longitudes of the flightpath.
    mask : xr.DataArray
        Mask of land and sea. 1 for sea, 0 for land.

    Returns
    -------
    xr.Dataset
        Radiometer data filtered for height and roll angle.
    """

    return (
        ds.pipe(_altitude_filter, height)
        .pipe(_roll_filter, roll)
        .pipe(_trim_dataset)
        .pipe(_filter_land, mask, lat, lon)
    )


def filter_iwv(ds):
    """Filter IWV data for spikes.

    Parameters
    ----------
    ds : xr.Dataset
        Level0 IWV dataset.

    Returns
    -------
    xr.Dataset
        IWV data filtered for spikes.
    """

    return ds.pipe(_filter_spikes)


def correct_radar_height(ds, roll, pitch, altitude):
    """Correct radar range gates with HALO flight altitude to height above WGS84 ellipsoid.

    Parameters
    ----------
    ds : xr.Dataset
        Level0 radar dataset.
    roll : xr.DataArray
        Flight roll angle in degrees.
    pitch : xr.DataArray
        Flight pitch angle in degrees.
    altitude : xr.DataArray
        Flight altitude in m.

    Returns
    -------
    xr.Dataset
        Radar data corrected to height above WGS84 ellipsoid.

    """
    z_grid = np.arange(0, altitude.max() + 30, 30)
    altitude_subsampled = altitude.sel(time=ds.time, method="nearest").assign_coords(
        time=ds.time
    )
    roll_subsampled = roll.sel(time=ds.time, method="nearest").assign_coords(
        time=ds.time
    )
    pitch_subsampled = pitch.sel(time=ds.time, method="nearest").assign_coords(
        time=ds.time
    )
    flight_los = (
        altitude_subsampled
        / np.cos(np.radians(pitch_subsampled))
        / np.cos(np.radians(roll_subsampled))
    )

    ds_z_grid = xr.DataArray(
        data=np.tile(z_grid, (len(ds.time), 1)),
        dims=("time", "height"),
        coords={"time": ds.time, "height": z_grid},
    )
    ds_range = xr.DataArray(
        coords={"time": ds.time, "height": z_grid},
        dims=("time", "height"),
        data=ds_z_grid
        / np.cos(np.radians(pitch_subsampled))
        / np.cos(np.radians(roll_subsampled)),
    )

    return ds.sel(range=flight_los - ds_range, method="nearest").where(
        ds_z_grid < altitude_subsampled
    )
