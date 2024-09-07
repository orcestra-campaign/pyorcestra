#!/usr/bin/env python3
import argparse

import cmocean as cmocean
import matplotlib.pyplot as plt
import numpy as np


def mpl2rgba(mpl):
    return [int(f * 255) for f in mpl]


def _cmap2windy_rgba(name, vmin, vmax, num=32):
    cmap = plt.get_cmap(name)
    norm = plt.Normalize(vmin=vmin, vmax=vmax)

    return [[float(v), mpl2rgba(cmap(norm(v)))] for v in np.linspace(vmin, vmax, num)]


def cmap2windy():
    parser = argparse.ArgumentParser(prog="matplotlib2windy")
    parser.add_argument("min", help="lower limit of color scale", type=float)
    parser.add_argument("max", help="upper limit of color scale", type=float)
    parser.add_argument("-c", "--cmap", help="color map", type=str, default="viridis")

    args = parser.parse_args()

    print(_cmap2windy_rgba(args.cmap, args.min, args.max))
