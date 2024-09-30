import xarray as xr


def get_meteor_track(deduplicate_latlon=False):
    """Load the METEOR track from DShip data and return as xr.Dataset.

    Args:
      deduplicate_latlon (bool): Deprecated.
    """
    dship = xr.open_zarr(
        "ipns://latest.orcestra-campaign.org/products/METEOR/DShip.zarr"
    )

    return dship[["lat", "lon"]]
