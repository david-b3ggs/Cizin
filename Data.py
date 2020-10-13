from openeew.data.aws import AwsDataClient
from openeew.data.df import get_df_from_records
import plotnine as pn
from datetime import datetime

def plot_seismograms(device_id):
    # Get earthquake date as datetime.datetime object
    eq_dt = AwsDataClient._get_dt_from_str(eq['date_utc'])

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
                xintercept=eq_dt,
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


data_client = AwsDataClient('mx')

eq = {
        'latitude': 16.218,
        'longitude': -98.0135,
        'date_utc': '2018-02-16 23:39:39'
        }

devices = data_client.get_devices_as_of_date(eq['date_utc'])

start_date_utc = '2018-02-16 23:39:00'
end_date_utc = '2018-02-16 23:43:00'

# Get records for the specified dates
records_df = get_df_from_records(
    data_client.get_filtered_records(
        start_date_utc,
        end_date_utc
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

plot_seismograms('011')


