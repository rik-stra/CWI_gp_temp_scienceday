import numpy as np

def get_data_point(lon, lat, data_array, min_lon, max_lon, min_lat, max_lat):
    """
    Get the data point from the data_array corresponding to the given lon and lat.
    """
    lon = np.clip(lon, min_lon, max_lon)
    lat = np.clip(lat, min_lat, max_lat)
    lon_idx = round((lon - min_lon) / (max_lon - min_lon) * (data_array.shape[1] - 1))
    lat_idx = round((lat - min_lat) / (max_lat - min_lat) * (data_array.shape[0] - 1))
    return data_array[lat_idx, lon_idx].item()



