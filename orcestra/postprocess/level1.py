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


def _roll_filter(ds, ds_bahamas):
    """Filter any dataset by plane roll angle.

    Parameters
    ----------
    ds : xr.Dataset
        Level0 dataset.
    ds_bahamas : xr.Dataset
        Level0 BAHAMAS dataset.

    Returns
    -------
    xr.Dataset
        Dataset filtered by plane roll angle.
    """

    ds_bahamas_subsampled = ds_bahamas.sel(
        time=ds.time, method="nearest"
    ).assign_coords(time=ds.time)

    return ds.where(np.abs(ds_bahamas_subsampled.IRS_PHI) < config["roll_threshold"])


def _altitude_filter(ds, ds_bahamas):
    """Filter any dataset by plane altitude.

    Parameters
    ----------
    ds : xr.Dataset
        Level0 dataset.
    ds_bahamas : xr.Dataset
        Level0 BAHAMAS dataset.

    Returns
    -------
    xr.Dataset
        Dataset filtered by plane altitude.
    """

    ds_bahamas_subsampled = ds_bahamas.sel(
        time=ds.time, method="nearest"
    ).assign_coords(time=ds.time)

    return ds.where(ds_bahamas_subsampled.IRS_ALT > config["altitude_threshold"])


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


def resample_radiometer(ds, freq=pd.Timedelta("1s")):
    """
    Resample radiometer data to a given frequency using pandas dataframe resample method to speed up the process.

    Parameters
    ----------
    ds : xarray.Dataset
        level 0 radiometer data
    freq : pandas.Timedelta
        resampling frequency, default is 1 second

    Returns
    -------
    xarray.Dataset
        resampled radiometer data
    """

    TBs = [
        ds.TBs.sel(frequency=f)
        .to_dataframe()
        .resample(freq)
        .mean()
        .to_xarray()
        .drop_vars("frequency")
        .expand_dims(dim={"frequency": pd.Index([f], name="frequency")})
        for f in ds.frequency
    ]
    ds_resampled = xr.concat(TBs, dim="frequency")
    freq_less_vars = [
        "ultra_sampling_factor",
        "rain_flag",
        "elevation_angle",
        "azimuth_angle",
    ]
    ds_resampled = ds_resampled.assign(
        ds[freq_less_vars].to_dataframe().resample(freq).mean().to_xarray()
    )
    return ds_resampled


def filter_radar(ds, ds_bahamas):
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
        .pipe(_roll_filter, ds_bahamas)
        .dropna(dim="time", how="all")
    )


def filter_radiometer(ds, ds_bahamas):
    """Filter radiometer data for height and roll angle.

    Parameters
    ----------
    ds : xr.Dataset
        Level0 radiometer dataset.

    Returns
    -------
    xr.Dataset
        Radiometer data filtered for height and roll angle.
    """

    return (
        ds.pipe(_altitude_filter, ds_bahamas)
        .pipe(_roll_filter, ds_bahamas)
        .dropna(dim="time", how="all")
    )


def correct_radar_height(ds, ds_bahamas):
    """Correct radar range gates with HALO flight altitude to height above WGS84 ellipsoid.

    Parameters
    ----------
    ds : xr.Dataset
        Level0 radar dataset.
    ds_bahamas : xr.Dataset
        Level0 BAHAMAS dataset.

    Returns
    -------
    xr.Dataset
        Radar data corrected to height above WGS84 ellipsoid.

    """
    z_grid = np.arange(0, ds_bahamas.IRS_ALT.max() + 30, 30)
    ds_bahamas_subsampled = ds_bahamas.sel(
        time=ds.time, method="nearest"
    ).assign_coords(time=ds.time)
    flight_los = (
        ds_bahamas_subsampled.IRS_ALT
        / np.cos(np.radians(ds_bahamas_subsampled.IRS_THE))
        / np.cos(np.radians(ds_bahamas_subsampled.IRS_PHI))
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
        / np.cos(np.radians(ds_bahamas_subsampled.IRS_THE))
        / np.cos(np.radians(ds_bahamas_subsampled.IRS_PHI)),
    )

    return ds.sel(range=flight_los - ds_range, method="nearest")
