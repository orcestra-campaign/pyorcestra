import numpy as np
import xarray as xr
import json
import os

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


def coarsen_radiometer(ds):
    """Coarsen radiometer data to 1 Hz.

    Parameters
    ----------
    ds : xr.Dataset
        Level0 radiometer dataset.

    Returns
    -------
    xr.Dataset
        Radiometer data coarsened to 1 Hz.
    """

    return ds.coarsen(time=4, boundary="pad").mean().drop_duplicates("time")


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


def filter_radiometer(ds, height, roll):
    """Filter radiometer data for height and roll angle.

    Parameters
    ----------
    ds : xr.Dataset
        Level0 radiometer dataset.
    height : xr.DataArray
        Flight altitude in m.
    roll : xr.DataArray
        Flight roll angle in degrees.

    Returns
    -------
    xr.Dataset
        Radiometer data filtered for height and roll angle.
    """

    return (
        ds.pipe(_altitude_filter, height).pipe(_roll_filter, roll).pipe(_trim_dataset)
    )


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
