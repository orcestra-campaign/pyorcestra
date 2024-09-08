from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike

KT2MS = 1852 / 3600


@dataclass
class FlSpeedPerformance:
    fls: ArrayLike  # Flight Level Units (i.e. 100ft)
    speeds: ArrayLike  # m/s

    def __post_init__(self):
        assert len(self.fls) == len(self.speeds)

    def speed_at_fl(self, fl):
        return np.interp(fl, self.fls, self.speeds)


aircraft_performance = {
    "HALO": FlSpeedPerformance(
        fls=np.array([190, 230, 260, 280, 330, 360, 390, 410, 430, 450, 470, 490]),
        speeds=np.array(
            [
                351.3,
                380.8,
                399.3,
                410.0,
                432.0,
                442.3,
                450.8,
                455.6,
                459.8,
                463.5,
                466.9,
                469.9,
            ]
        )
        * KT2MS,
    ),
}

DEFAULT_PERFORMANCE = aircraft_performance["HALO"]
CURRENT_PERFORMANCE = None


def get_current_performance():
    return CURRENT_PERFORMANCE or DEFAULT_PERFORMANCE


def set_current_performance(name_or_performance):
    global CURRENT_PERFORMANCE
    if isinstance(name_or_performance, str):
        return set_current_performance(aircraft_performance[name_or_performance])
    CURRENT_PERFORMANCE = name_or_performance


def get_flight_performance(aircraft=None):
    if aircraft is None:
        return get_current_performance()
    else:
        return aircraft_performance[aircraft]
