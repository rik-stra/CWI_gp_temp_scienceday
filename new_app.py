from weakref import ref
import streamlit as st
import folium
from streamlit_folium import st_folium
from folium.plugins import MarkerCluster
import numpy as np
import branca.colormap as cm
from netCDF4 import Dataset
from gp_app.GP_model import fit_gp_model, predict_gp_model, normalize_for_train, scale_for_predict, rescale_output
from gp_app.map_tools import get_data_point
import os
import matplotlib.pyplot as plt

map_size = (700, 800)

modi = ["kies eigen meetpunten", "KNMI meetpunten"]
data = ["14 juli 2025", "1 juli 2025", "1 januari 2025", "18 februari 2025"]

knmi_station_lats = [53.124, 52.457, 53.223, 52.055, 52.14 , 52.643, 52.435, 51.961,
       51.565, 52.099, 51.448, 51.441, 52.317, 52.505, 52.702, 53.194,
       51.991, 50.905, 52.897, 52.273, 51.45 , 51.197, 53.24 , 53.412,
       52.927, 52.749, 51.225, 51.858, 51.497, 53.391, 52.068, 51.969,
       51.659]
knmi_station_lons = [6.585, 5.52 , 5.752, 5.872, 4.436, 4.979, 6.259, 4.447, 4.935,
       5.18 , 4.342, 3.596, 4.79 , 4.603, 5.887, 7.149, 4.122, 5.762,
       5.383, 6.891, 5.377, 5.763, 4.921, 6.199, 4.781, 6.573, 3.861,
       5.145, 6.196, 5.346, 6.657, 4.926, 5.707]

date_dict = {"14 juli 2025":          "data/high_resolution_weather_maps_202507141150.nc",
                "1 juli 2025":        "data/high_resolution_weather_maps_202507011620.nc",
                "1 januari 2025":     "data/high_resolution_weather_maps_202501010010.nc",
                "18 februari 2025":   "data/high_resolution_weather_maps_202502180810.nc"}

# if folder does not exist create folder for highscores
if not os.path.exists("./high_scores"):
    os.makedirs("./high_scores")
    for date in date_dict.keys():
        scores = np.ones(50)*20.0
        np.save(f"./high_scores/high_scores_{date.replace(' ', '_')}.npy", scores)

st.session_state.mode = st.sidebar.selectbox("Selecteer modus", modi)
st.session_state.datum = st.sidebar.selectbox("Selecteer datum", data)
if "KNMI_on_map" not in st.session_state:
    st.session_state.KNMI_on_map = False
if "reset_needed" not in st.session_state:
    st.session_state.reset_needed = False
if "current_date" not in st.session_state:
    st.session_state.current_date = "lala"


# get ref data
if not (st.session_state.datum == st.session_state.current_date):
    st.session_state.current_date = st.session_state.datum
    st.session_state.reset_needed = True
    rootgrp = Dataset(date_dict[st.session_state.datum], "r", format="NETCDF4")
    st.session_state.ref_temperature = rootgrp.variables['mean_DryBulbTemperature_Celsius'][0]
    st.session_state.ref_longitude = rootgrp.variables['Longitude'][0]
    st.session_state.ref_latitude = rootgrp.variables['Latitude'][0]
    st.session_state.ref_water_fraction = rootgrp.variables['cov_water_fraction'][0]
    st.session_state.mask = np.array(rootgrp.variables['cov_mask'][0])
    st.session_state.min_lon, st.session_state.max_lon = st.session_state.ref_longitude.min(), st.session_state.ref_longitude.max()
    st.session_state.min_lat, st.session_state.max_lat = st.session_state.ref_latitude.min(), st.session_state.ref_latitude.max()

    linear_cm_temp = cm.LinearColormap(["blue", "yellow", "red"], vmin=st.session_state.ref_temperature.min()-5, vmax=st.session_state.ref_temperature.max()+5)

    st.session_state.score = np.ones(50)*20.0
    st.session_state.high_scores = np.load(f"./high_scores/high_scores_{st.session_state.datum.replace(' ', '_')}.npy")

if "center" not in st.session_state:
    st.session_state.center = [52.2, 5.28]  # Default location
    st.session_state.zoom = 7.5  # Default zoom

if st.sidebar.button("Reset", width="stretch") or st.session_state.reset_needed:
    st.session_state.reset_needed = False
    lat = 52.801035484049724
    lon = 5.177993774414063
    st.session_state.station_number = [0]
    st.session_state.station_lat = [lat]
    st.session_state.station_lon = [lon]
    st.session_state.temperature = [get_data_point(lon, lat, st.session_state.ref_temperature, st.session_state.min_lon, st.session_state.max_lon, st.session_state.min_lat, st.session_state.max_lat)]
    st.session_state.water_fraction = [get_data_point(lon, lat, st.session_state.ref_water_fraction, st.session_state.min_lon, st.session_state.max_lon, st.session_state.min_lat, st.session_state.max_lat)]
    st.session_state.plot_type = "Temp"
    st.session_state.refit = True
    st.session_state.KNMI_on_map = False
    if "GP_fields" in st.session_state:
        del st.session_state.GP_fields
    if "error" in st.session_state:
        del st.session_state.error

if len(st.session_state.station_number) > 50:
    st.warning("De thermometers zijn op, reset de kaart om opnieuw te beginnen.")
    st.stop()


if st.session_state.mode == "KNMI meetpunten":
    if not st.session_state.KNMI_on_map:
        for i in range(len(knmi_station_lats)):
            st.session_state.station_number.append(len(st.session_state.station_number))
            st.session_state.station_lat.append(knmi_station_lats[i])
            st.session_state.station_lon.append(knmi_station_lons[i])
            st.session_state.temperature.append(get_data_point(knmi_station_lons[i], knmi_station_lats[i], st.session_state.ref_temperature, st.session_state.min_lon, st.session_state.max_lon, st.session_state.min_lat, st.session_state.max_lat))
            st.session_state.water_fraction.append(get_data_point(knmi_station_lons[i], knmi_station_lats[i], st.session_state.ref_water_fraction, st.session_state.min_lon, st.session_state.max_lon, st.session_state.min_lat, st.session_state.max_lat))
        st.session_state.KNMI_on_map = True
        st.session_state.refit = True
    

left, middle_left, middle_right = st.columns(3)
if left.button("Temp", width="stretch"):
    st.session_state.plot_type = "Temp"
if middle_left.button("Onzekerheid", width="stretch"):
    st.session_state.plot_type = "Onzekerheid"
if middle_right.button("Fout", width="stretch"):
    st.session_state.plot_type = "Fout"

if st.sidebar.button("Cancel laatste meetpunt", width="stretch"):
    if len(st.session_state.station_number) > 1:
        st.session_state.station_number.pop()
        st.session_state.station_lat.pop()
        st.session_state.station_lon.pop()
        st.session_state.temperature.pop()
        st.session_state.water_fraction.pop()
        st.session_state.refit = True



# Create the base map
m = folium.Map(location=st.session_state.center, zoom_start=st.session_state.zoom, tiles="Cartodb Positron")
marker_cluster = MarkerCluster().add_to(m)
for i in range(len(st.session_state.station_number)):
    folium.Marker([st.session_state.station_lat[i], st.session_state.station_lon[i]], popup=f'{st.session_state.temperature[i]:.1f} °C').add_to(marker_cluster)

if "GP_fields" in st.session_state:
        if st.session_state.plot_type == "Temp":
            gp_field = st.session_state.GP_fields[0]
            ticks = [i for i in range(int(np.floor(gp_field.min())), int(np.ceil(gp_field.max())))]
            linear_cm = cm.LinearColormap(["blue", "yellow", "red"], vmin=gp_field.min()-1, vmax=gp_field.max()+1,
                                          tick_labels=ticks,
                                           caption='Temperatuur (°C)')
        elif st.session_state.plot_type == "Onzekerheid":
            gp_field = st.session_state.GP_fields[1]
            ticks = [i for i in range(int(np.floor(gp_field.min())), int(np.ceil(gp_field.max())))]
            linear_cm = cm.LinearColormap(["blue", "red"], vmin=gp_field.min(), vmax=gp_field.max(),
                                           tick_labels=ticks,
                                           caption='Onzekerheid (°C)')
        else:
            gp_field = np.multiply(st.session_state.mask,np.abs(st.session_state.GP_fields[0]-st.session_state.ref_temperature))
            ticks = [i for i in range(int(np.floor(gp_field.min())), int(np.ceil(gp_field.max())))]
            linear_cm = cm.LinearColormap(["blue", "red"], vmin=gp_field.min(), vmax=gp_field.max(),
                                            tick_labels=ticks,
                                           caption='Fout (°C)')
        image = folium.raster_layers.ImageOverlay(
            image=np.array(gp_field),
            bounds=[[st.session_state.min_lat, st.session_state.min_lon], [st.session_state.max_lat, st.session_state.max_lon]],
            opacity=0.6,
            interactive=True,
            origin="lower",
            colormap=linear_cm,
            mercator_project=True,
        )
        image.add_to(m)
        linear_cm.add_to(m)
else:
    linear_cm = cm.LinearColormap(["blue", "yellow", "red"], vmin=10, vmax=20,
                                           caption='Temperatuur (°C)')
    linear_cm.add_to(m)
        

if st.session_state.refit and len(st.session_state.station_number) > 1:
        input, output, scaling = normalize_for_train([st.session_state.station_lat, st.session_state.station_lon, st.session_state.water_fraction], st.session_state.temperature)
        model = fit_gp_model(input, output, n_train=50)

        input_pred = scale_for_predict([st.session_state.ref_latitude.flatten(), st.session_state.ref_longitude.flatten(), st.session_state.ref_water_fraction.flatten()], scaling)
        mean, std = predict_gp_model(model, input_pred)
        mean = rescale_output(mean, scaling).cpu().numpy().reshape(st.session_state.ref_latitude.shape)
        std = std * scaling['std_out']
        std = std.cpu().numpy().reshape(st.session_state.ref_latitude.shape)
        st.session_state.GP_fields = [mean, np.multiply(std,st.session_state.mask)]
        st.session_state.error = [np.max(np.abs(np.multiply(st.session_state.mask,st.session_state.ref_temperature) - np.multiply(st.session_state.mask,mean))), 
                                  np.sum(np.abs(np.multiply(st.session_state.mask,st.session_state.ref_temperature) - np.multiply(st.session_state.mask,mean)))/np.sum(st.session_state.mask)]
        st.session_state.refit = False
    # Add raster overlay

# Render the map and capture clicks
map = st_folium(m, width=map_size[0], height=map_size[1], key="folium_map")

if "error" in st.session_state:
    st.write(f'Aantal meetpunten: {len(st.session_state.station_number)-1:<10} - Gemiddelde fout: {st.session_state.error[1]:.2f} celsius - Maximale fout: {st.session_state.error[0]:.2f} celsius')
    st.session_state.score[len(st.session_state.station_number)-2] = st.session_state.error[1]

with st.sidebar:
    n_station = len(st.session_state.station_number)-1
    x = np.arange(1, 51)
    x_own = np.arange(1, n_station+1)

    fig, ax = plt.subplots()
    index = (st.session_state.high_scores<19)
    if sum(index)>0:
        ax.plot(x[index],st.session_state.high_scores[index], label="High scores", color='gold')
    if n_station > 0:
        ax.plot(x_own[(st.session_state.score[:n_station]<19)],st.session_state.score[:n_station][st.session_state.score[:n_station]<19], label="Jouw score", color='green')
    ax.set_title("Score")
    ax.set_xlabel("Meetpunten")
    ax.set_ylabel("Gemiddelde fout")
    ax.legend()
    st.pyplot(fig)

if st.sidebar.button("Save scores", width="stretch"):
    st.session_state.high_scores = np.minimum(st.session_state.high_scores, st.session_state.score)
    np.save(f"./high_scores/high_scores_{st.session_state.datum.replace(' ', '_')}.npy", st.session_state.high_scores)
    st.sidebar.success("Scores opgeslagen!")

# Update marker position immediately after each click
if map.get("last_clicked"):
    lat, lng = map["last_clicked"]["lat"], map["last_clicked"]["lng"]
    lat = np.clip(lat, st.session_state.min_lat, st.session_state.max_lat)
    lng = np.clip(lng, st.session_state.min_lon, st.session_state.max_lon)
    st.session_state.station_number.append(len(st.session_state.station_number))
    st.session_state.station_lat.append(lat)
    st.session_state.station_lon.append(lng)
    st.session_state.temperature.append(get_data_point(lng, lat, st.session_state.ref_temperature, st.session_state.min_lon, st.session_state.max_lon, st.session_state.min_lat, st.session_state.max_lat))
    st.session_state.water_fraction.append(get_data_point(lng, lat, st.session_state.ref_water_fraction, st.session_state.min_lon, st.session_state.max_lon, st.session_state.min_lat, st.session_state.max_lat))
    # Redraw the map immediately with the new marker location
    
    m = folium.Map(location=st.session_state.center, zoom_start=st.session_state.zoom, tiles="Cartodb Positron")
    marker_cluster = MarkerCluster().add_to(m)
    for i in range(len(st.session_state.station_number)):
        folium.Marker([st.session_state.station_lat[i], st.session_state.station_lon[i]], popup=f'{st.session_state.temperature[i]} °C').add_to(marker_cluster)
    
    map = st_folium(m, width=map_size[0], height=map_size[1], key="folium_map")
    st.session_state.refit = True
    

