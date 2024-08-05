import numpy as np


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


def radar_ql(ds):
    """Post-processing of Radar quick look datasets."""
    return ds.pipe(
        _radar_ql_fix_time,
    ).pipe(
        _radar_ql_add_dBZ,
    )


def radiometer(ds):
    """Post-processing of radiometer datasets."""
    return ds.rename(number_frequencies="frequency").set_index(frequency="frequencies")
