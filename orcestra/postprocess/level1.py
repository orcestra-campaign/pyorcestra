import numpy as np
import pandas as pd


def _bahamas_fix_time(ds):
    """Fix time coordinate of BAHAMAS datasets."""
    return ds.rename({"tid": "time", "TIME": "time"}).set_index(time="time")


def _radar_fix_time(ds):
    """Fix time coordinate of RADAR moments datasets."""
    datetime = (
        np.datetime64("1970-01-01", "ns")
        + ds.time.values * np.timedelta64(1, "s")
        + ds.microsec.values * np.timedelta64(1, "us")
    )

    return ds.assign(time=datetime).drop_vars("microsec")


def _radar_add_dBZ(ds):
    """Add reflectivity in dB."""
    ds = ds.assign(
        dBZg=lambda dx: 10 * np.log10(dx.Zg),
        dBZe=lambda dx: 10 * np.log10(dx.Ze),
    )
    ds.dBZg.attrs = {
        "units": "dBZg",
        "long_name": "Decadal logarithm of equivalent radar reflectivity of all targets (Zg)",
    }
    ds.dBZe.attrs = (
        {
            "units": "dBZe",
            "long_name": "Decadal logarithm of equivalent radar reflectivity of hydrometeors (Ze)",
        },
    )

    return ds


def _fix_radiometer_time(ds):
    """Replace duplicates in time coordinate of radiometer datasets with correct time and ensure 4Hz frequency."""

    # replace duplicate values
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

    ds = ds.assign_coords(time=time_new).sortby("time").drop_duplicates("time")

    # ensure 4Hz frequency
    start_time = ds.time.min().values
    end_time = ds.time.max().values
    time_expected = pd.date_range(start=start_time, end=end_time, freq="0.25s")
    ds = ds.reindex(time=time_expected, fill_value=np.nan)

    return ds


def _add_georeference(ds, ds_bahamas):
    """Add georeference information to dataset."""
    return ds.assign(
        plane_altitude=ds_bahamas.IRS_ALT.sel(
            time=ds.time, method="nearest"
        ).assign_coords(time=ds.time),
        lat=ds_bahamas.IRS_LAT.sel(time=ds.time, method="nearest").assign_coords(
            time=ds.time
        ),
        lon=ds_bahamas.IRS_LON.sel(time=ds.time, method="nearest").assign_coords(
            time=ds.time
        ),
        plane_roll=ds_bahamas.IRS_PHI.sel(time=ds.time, method="nearest").assign_coords(
            time=ds.time
        ),
        plane_pitch=ds_bahamas.IRS_THE.sel(
            time=ds.time, method="nearest"
        ).assign_coords(time=ds.time),
    )


def fix_bahamas(ds):
    """Post-processing of BAHAMAS datasets."""
    return ds.pipe(
        _bahamas_fix_time,
    )


def fix_radiometer(ds, ds_bahamas):
    """Post-processing of radiometer datasets."""
    return (
        ds.rename(number_frequencies="frequency")
        .set_index(frequency="Freq")
        .pipe(
            _fix_radiometer_time,
        )
        .pipe(
            _add_georeference,
            ds_bahamas,
        )
    )


def fix_iwv(ds, ds_bahamas):
    """Post-processing of IWV datasets."""
    return ds.pipe(
        _fix_radiometer_time,
    ).pipe(_add_georeference, ds_bahamas)


def fix_radar(ds, ds_bahamas):
    """Post-processing of Radar quick look datasets."""
    return (
        ds.pipe(
            _radar_fix_time,
        )
        .pipe(
            _radar_add_dBZ,
        )
        .pipe(
            _add_georeference,
            ds_bahamas,
        )
    )
