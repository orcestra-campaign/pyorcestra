import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import xarray as xr


def plot_sat(
    date_time_obj,
    layer="MODIS_Terra_CorrectedReflectance_TrueColor,Coastlines_15m",
    resolution=0.05,  # in degree
    bbox=[-10.0, -65.0, 25.0, -5.0],  # S, W, N, E
):
    from io import BytesIO
    import requests

    time = date_time_obj.strftime("%Y-%m-%dT%H:%M:%SZ")

    bbox_str = ",".join([str(v) for v in bbox])
    img_format = "png"

    lat_min, lon_min, lat_max, lon_max = bbox

    width, height = (
        int((lon_max - lon_min) / resolution),
        int((lat_max - lat_min) / resolution),
    )

    url = f"https://wvs.earthdata.nasa.gov/api/v1/snapshot?REQUEST=GetSnapshot&TIME={time}&BBOX={bbox_str}&CRS=EPSG:4326&LAYERS={layer}&FORMAT=image/{img_format}&WIDTH={width}&HEIGHT={height}"

    r = requests.get(url)

    img = plt.imread(BytesIO(r.content))
    plt.imshow(
        img,
        origin="upper",
        extent=[lon_min, lon_max, lat_min, lat_max],
        transform=ccrs.PlateCarree(),
    )


def _create_GOES_variable(goes_object: xr.Dataset, variable: str, gamma: float = 1.2):
    """Create a GOES image that can be plotted with `plt.imshow()`."""
    GOES_PRODUCT_DICT = {
        "AirMass": goes_object.rgb.AirMass,
        "AirMassTropical": goes_object.rgb.AirMassTropical,
        "AirMassTropicalPac": goes_object.rgb.AirMassTropicalPac,
        "Ash": goes_object.rgb.Ash,
        "DayCloudConvection": goes_object.rgb.DayCloudConvection,
        "DayCloudPhase": goes_object.rgb.DayCloudPhase,
        "DayConvection": goes_object.rgb.DayConvection,
        "DayLandCloud": goes_object.rgb.DayLandCloud,
        "DayLandCloudFire": goes_object.rgb.DayLandCloudFire,
        "DaySnowFog": goes_object.rgb.DaySnowFog,
        "DifferentialWaterVapor": goes_object.rgb.DifferentialWaterVapor,
        "Dust": goes_object.rgb.Dust,
        "FireTemperature": goes_object.rgb.FireTemperature,
        "NaturalColor": goes_object.rgb.NaturalColor(gamma=gamma),
        "NightFogDifference": goes_object.rgb.NightFogDifference,
        "NighttimeMicrophysics": goes_object.rgb.NighttimeMicrophysics,
        "NormalizedBurnRatio": goes_object.rgb.NormalizedBurnRatio,
        "RocketPlume": goes_object.rgb.RocketPlume,
        "SeaSpray": goes_object.rgb.SeaSpray(gamma=gamma),
        "SplitWindowDifference": goes_object.rgb.SplitWindowDifference,
        "SulfurDioxide": goes_object.rgb.SulfurDioxide,
        "TrueColor": goes_object.rgb.TrueColor(gamma=gamma),
        "WaterVapor": goes_object.rgb.WaterVapor,
    }
    return GOES_PRODUCT_DICT[variable]


def goes_overlay(
    image_date, ax, satellite="16", product="ABI", domain="F", variable="TrueColor"
):
    from goes2go.data import goes_nearesttime

    snapshot = goes_nearesttime(
        image_date, satellite=satellite, product=product, domain=domain
    )
    ax.imshow(
        _create_GOES_variable(snapshot, variable),
        transform=snapshot.rgb.crs,
        regrid_shape=3500,
        interpolation="nearest",
    )
    return


def dropsondes_overlay(
    dropsonde_ds,
    ax,
    variable="iwv",
    variable_label=r"Integrated Water Vapor / kg m$^{-2}$",
    colormap="Blues_r",
    markershape="o",
    markersize=40,
    edgecolor="grey",
    vmin=45,
    vmax=70,
):
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    im_launches = ax.scatter(
        dropsonde_ds["lon"].isel(alt=10),
        dropsonde_ds["lat"].isel(alt=10),
        marker=markershape,
        edgecolor=edgecolor,
        s=markersize,
        transform=ccrs.PlateCarree(),
        c=dropsonde_ds[variable],
        cmap=colormap,
        vmin=vmin,
        vmax=vmax,
    )

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("bottom", size="3%", pad=0.4, axes_class=plt.Axes)
    cbar = plt.colorbar(im_launches, cax=cax, orientation="horizontal")
    cbar.set_label(variable_label)

    return im_launches
