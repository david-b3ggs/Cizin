import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from openeew.data.aws import AwsDataClient
import folium

data_client = AwsDataClient('mx')

df = pd.read_csv("./quakeList.csv")

eq = []
for x in df.index:
    eq.append({
        'latitude': df["latitude"][x],
        'longitude': df["longitude"][x],
        'time': df["date"][x] + " " + df["time"][x]
    })


m = folium.Map(
            location=[eq[0]['latitude'], eq[0]['longitude']],
            zoom_start=7,
            titles="Signifigant Quakes After 2018"
            )

for point in eq:
    folium.Circle(
        radius=10000,
        location=[point['latitude'], point['longitude']],
        color='crimson',
        fill='crimson',
    ).add_to(m)

for point in eq:
        folium.Marker(
            [point['latitude'], point['longitude']],
            popup = folium.Popup(
                point['time'],
                sticky=True
                ),
            icon = folium.Icon(color="red", icon="glyphicon-flash")
            ).add_to(m)

devices = data_client.get_devices_as_of_date(eq[0]['time'])

for d in devices:
        folium.Marker(
            [d['latitude'], d['longitude']],
            popup = folium.Popup(
                d['device_id'],
                sticky=True
                )
            ).add_to(m)
m.save('QUAKES.html')


