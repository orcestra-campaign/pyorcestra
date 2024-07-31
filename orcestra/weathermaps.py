import cartopy.crs as ccrs
import matplotlib.pyplot as plt


def plot_sat(date_time_obj, 
             layer = "MODIS_Terra_CorrectedReflectance_TrueColor,Coastlines_15m",
             resolution = 0.05,  # in degree
             bbox = [-10.0, -65.0, 25.0, -5.0]  # S, W, N, E
            ):
    
    from io import BytesIO
    import requests
    
    time = date_time_obj.strftime("%Y-%m-%dT%H:%M:%SZ")
    
    bbox_str = ",".join([str(v) for v in bbox])
    img_format = "png"
    
    lat_min, lon_min, lat_max, lon_max = bbox
    
    width, height = int((lon_max - lon_min) / resolution), int((lat_max - lat_min) / resolution)

    url = f"https://wvs.earthdata.nasa.gov/api/v1/snapshot?REQUEST=GetSnapshot&TIME={time}&BBOX={bbox_str}&CRS=EPSG:4326&LAYERS={layer}&FORMAT=image/{img_format}&WIDTH={width}&HEIGHT={height}"
    
    r = requests.get(url)

    img = plt.imread(BytesIO(r.content))
    plt.imshow(img, 
               origin='upper', 
               extent=[lon_min, lon_max, lat_min, lat_max], 
               transform=ccrs.PlateCarree())
