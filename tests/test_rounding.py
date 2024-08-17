from orcestra.flightplan import LatLon
import pytest


TEST_CASES = [
    (
        LatLon(13.9999, 0),
        "1400N00000E",
        "1400N00000E",
        "1359.99N00000.00E",
        "N13 59.99, E000 00.00",
    ),
    (
        LatLon(13.99999, 0),
        "1400N00000E",
        "1400N00000E",
        "1400.00N00000.00E",
        "N14 00.00, E000 00.00",
    ),
    (
        LatLon(5.0, -26.996),
        "0500N02700W",
        "0500N02700W",
        "0500.00N02659.76W",
        "N05 00.00, W026 59.76",
    ),
    (
        LatLon(-2.05, -7.84),
        "0205S00750W",
        "0203S00750W",
        "0203.00S00750.40W",
        "S02 03.00, W007 50.40",
    ),
]

TC_5MIN, TC_1MIN, TC_MIN100, TC_PILOT = [
    [(tc[0], tc[i + 1]) for tc in TEST_CASES] for i in range(4)
]


@pytest.mark.parametrize("ll,expected", TC_5MIN)
def test_5min(ll, expected):
    assert ll.format_5min() == expected


@pytest.mark.parametrize("ll,expected", TC_1MIN)
def test_1min(ll, expected):
    assert ll.format_1min() == expected


@pytest.mark.parametrize("ll,expected", TC_MIN100)
def test_min100(ll, expected):
    assert ll.format_min100() == expected


@pytest.mark.parametrize("ll,expected", TC_PILOT)
def test_centimin(ll, expected):
    assert ll.format_pilot() == expected
