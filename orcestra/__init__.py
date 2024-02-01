from dataclasses import dataclass


@dataclass
class LatLon:
    lat: float
    lon: float


bco = LatLon(13.079773, -59.487634)
sal = LatLon(16.73448797020352, -22.94397423993749)

__all__ = ["LatLon", "bco", "sal"]
