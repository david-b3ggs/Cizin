from openeew.data.aws import AwsDataClient
from openeew.data.df import get_df_from_records
import plotnine as pn
import datetime
import pandas as pd
from sklearn.preprocessing import QuantileTransformer
from sklearn import preprocessing
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

def plot_seismograms(device_id):
    # Get earthquake date as datetime.datetime object
    eq_dt = AwsDataClient._get_dt_from_str(eq['date_utc'])
    print(eq_dt)
    ob = {
        "ti" : "2018-02-16 23:39:48"
    }
    time_format = '%Y-%m-%d %H:%M:%S'
    plots = []
    for axis in ['x', 'y', 'z']:
        plots.append(
            pn.ggplot(
                records_df[records_df['device_id'] == device_id],
                pn.aes('sample_dt', axis)
            ) + \
            pn.geom_line(color='blue') + \
            pn.scales.scale_x_datetime(
                date_breaks='1 minute',
                date_labels='%H:%M:%S'
            ) + \
            pn.geoms.geom_vline(

                xintercept= eq_dt,#datetime.strptime(ob["ti"], time_format),
                color='crimson'
            ) + \
            pn.labels.ggtitle(
                'device {}, axis {}'.format(
                    device_id, axis)
            )
        )

    # Now output the plots
    for p in plots:
        print(p)


def getSensorData(quake, sensor):
    global eq, records_df
    data_client = AwsDataClient('mx')
    eq = quake
    devices = data_client.get_devices_as_of_date(eq['date_utc'])
    quakeTime = quake['date_utc']
    minute = timedelta(minutes=1)
    tenMinutes = timedelta(minutes=10)

    start_date_utc = datetime.fromisoformat(quakeTime) - minute
    end_date_utc = datetime.fromisoformat(quakeTime) + minute + minute + minute

    # Get records for the specified dates
    records_df = get_df_from_records(
        data_client.get_filtered_records(
            str(start_date_utc),
            str(end_date_utc)
        )
    )
    records_df['sample_dt'] = \
        records_df['sample_t'].apply(
            lambda x: datetime.utcfromtimestamp(x)
        )
    # Select required columns
    records_df = records_df[
        ['device_id', 'x', 'y', 'z', 'sample_dt']
    ]
    print(records_df.head())
    plot_seismograms(sensor)


quake = {
    'latitude': 16.218,
    'longitude': -98.0135,
    'date_utc': '2018-02-16 23:39:39'
}
sensor = '006'

getSensorData(quake, sensor)

"""df = pd.read_csv("classifyList.csv")
print(df.head())

data = df.values
scaler = preprocessing.MinMaxScaler()
scaled = scaler.fit_transform(data)

df = pd.DataFrame(scaled)
print(df.head())
ax1 = df.plot.scatter(x='x',
                      y='y',
                      c='z',
                      colormap='viridis')
df.plot()
plt.show()

# Plotting raw data before transformation
"""