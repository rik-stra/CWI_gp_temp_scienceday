import os
import numpy as np

date_dict = {"14 juli 2025":          "data/high_resolution_weather_maps_202507141150.nc",
                "1 juli 2025":        "data/high_resolution_weather_maps_202507011620.nc",
                "1 januari 2025":     "data/high_resolution_weather_maps_202501010010.nc",
                "18 februari 2025":   "data/high_resolution_weather_maps_202502180810.nc"}

# if folder does not exist create folder for highscores

for date in date_dict.keys():
    scores = np.ones(50)*20.0
    np.save(f"./high_scores/high_scores_{date.replace(' ', '_')}.npy", scores)