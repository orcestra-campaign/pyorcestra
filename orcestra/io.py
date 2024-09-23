import fsspec
import numpy as np
import xarray as xr


def _parse_igi(txtfile, skip_header, delimiter, varinfo, flight_date, gps_time_offset):
    """Parse IGI position data txt file and return as xr.Dataset."""
    with fsspec.open(txtfile) as fp:
        data = np.genfromtxt(fp, skip_header=skip_header, delimiter=delimiter)

    ds = xr.Dataset(
        data_vars={
            varname: xr.DataArray(data=data[:, i], attrs=attrs, dims=("time",))
            for i, (varname, attrs) in enumerate(varinfo.items())
        }
    )

    ds = ds.assign_coords(
        time=ds.time * np.timedelta64(1_000_000_000, "ns")
        + np.datetime64(flight_date)
        + gps_time_offset
    )

    return ds


def read_igi(txtfile, flight_date, gps_time_offset=np.timedelta64(-18, "s")):
    """Parse IGI position data txt file (1/10 Hz) and return as xr.Dataset."""
    _varinfo = {
        "time": dict(long_name="Generic/Time", unit="s"),
        "IRS_LAT": dict(long_name="WGS84 Datum/Latitude", unit="degrees_north"),
        "IRS_LON": dict(long_name="WGS84 Datum/Longitude", unit="degrees_east"),
        "IRS_ALT": dict(long_name="WGS84 Datum/Elliptical Height", unit="m"),
        "IRS_NSV": dict(long_name="Velocity/North", unit="m/s"),
        "IRS_EWV": dict(long_name="Velocity/East", unit="m/s"),
        "IRS_VV": dict(long_name="Velocity/Up", unit="m/s"),
        "IRS_PHI": dict(long_name="Attitude/Roll", unit="degree"),
        "IRS_THE": dict(long_name="Attitude/Pitch", unit="degree"),
        "IRS_R": dict(long_name="Attitude/Yaw", unit="degree"),
        "IGI_RMSX": dict(long_name="RMS/Position North", unit="m"),
        "IGI_RMSY": dict(long_name="RMS/Position East", unit="m"),
        "IGI_RMSZ": dict(long_name="RMS/Position Altitude", unit="m"),
    }

    return _parse_igi(
        txtfile,
        skip_header=83,
        delimiter=",",
        varinfo=_varinfo,
        flight_date=flight_date,
        gps_time_offset=gps_time_offset,
    )


def read_bahamas_100hz(txtfile, flight_date, gps_time_offset=np.timedelta64(-18, "s")):
    """Parse BAHAMAS 100 Hz position data txt file and return as xr.Dataset."""
    _varinfo = {
        "time": dict(long_name="Generic/Time", unit="s"),
        "IRS_LON": dict(long_name="WGS84 Datum/Longitude", unit="degrees_east"),
        "IRS_LAT": dict(long_name="WGS84 Datum/Latitude", unit="degrees_north"),
        "IRS_ALT": dict(long_name="WGS84 Datum/Elliptical Height", unit="m"),
        "IRS_PHI": dict(long_name="Attitude/Roll", unit="degree"),
        "IRS_THE": dict(long_name="Attitude/Pitch", unit="degree"),
        "IRS_R": dict(long_name="Attitude/Yaw", unit="degree"),
    }

    return _parse_igi(
        txtfile,
        skip_header=81,
        delimiter=None,
        varinfo=_varinfo,
        flight_date=flight_date,
        gps_time_offset=gps_time_offset,
    )
