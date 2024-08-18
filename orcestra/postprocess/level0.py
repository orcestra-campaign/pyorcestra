import numpy as np
import pandas as pd


def _bahamas_fix_time(ds):
    """Fix time coordinate of BAHAMAS datasets."""
    return ds.rename({"tid": "time", "TIME": "time"}).set_index(time="time")


def bahamas(ds):
    """Post-processing of BAHAMAS datasets."""
    return ds.pipe(
        _bahamas_fix_time,
    )


def _radar_ql_fix_time(ds):
    """Fix time coordinate of RADAR QL datasets."""
    datetime = (
        np.datetime64("1970-01-01", "ns")
        + ds.time.values * np.timedelta64(1, "s")
        + ds.microsec.values * np.timedelta64(1, "us")
    )

    return ds.assign(time=datetime).drop_vars("microsec")


def _radar_ql_add_dBZ(ds):
    """Add reflectivity in dB."""
    return ds.assign(
        dBZg=lambda dx: 10 * np.log10(dx.Zg),
        dBZe=lambda dx: 10 * np.log10(dx.Ze),
    )


def radar(ds):
    """Post-processing of Radar quick look datasets."""
    return ds.pipe(
        _radar_ql_fix_time,
    ).pipe(
        _radar_ql_add_dBZ,
    )


def _fix_radiometer_time(ds):
    """Replace duplicates in time coordinate of radiometer datasets with correct time"""
    time_broken = ds.time.values
    first_occurence = time_broken[0]
    n = 0
    time_new = []
    for i in range(len(time_broken)):
        if time_broken[i] == first_occurence:
            time_new.append(time_broken[i] + pd.Timedelta("0.25s") * n)
            n += 1
        else:
            n = 1
            first_occurence = time_broken[i]
            time_new.append(first_occurence)

    return ds.assign_coords(time=time_new).sortby("time")


def radiometer(ds):
    """Post-processing of radiometer datasets."""
    return (
        ds.rename(number_frequencies="frequency")
        .set_index(frequency="Freq")
        .pipe(
            _fix_radiometer_time,
        )
    )
