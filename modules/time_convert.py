import pytz
import datetime


def convert_to_local_time(open_time):
    timestamp = open_time / 1000
    dt = datetime.datetime.fromtimestamp(timestamp, tz=datetime.timezone.utc)
    tz = pytz.timezone('Asia/Taipei')
    localized_dt = dt.astimezone(tz)
    return localized_dt.strftime('%Y-%m-%d %H:%M')