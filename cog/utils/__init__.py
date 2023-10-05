import time
import pytz
from datetime import datetime

time_format = "%Y-%m-%dT%H:%M:%S.%fZ"


def convert_timestamp_to_datetime(timestamp):
    # Convert to a datetime object in UTC timezone
    utc_timezone = pytz.timezone("UTC")
    dt_object = datetime.fromtimestamp(timestamp, tz=utc_timezone)

    # Format the datetime object to the desired string format
    formatted_time = dt_object.strftime(time_format)
    return formatted_time


def convert_datetime_to_timestamp(datetime_str: str) -> float:
    return time.mktime(datetime.strptime(datetime_str, time_format).timetuple())


def measure_execution_time(func, *args, **kwargs):
    start_time = time.time()
    result = func(*args, **kwargs)
    end_time = time.time()
    elapsed_time = end_time - start_time
    return result, elapsed_time, start_time, end_time


def measure_execution_time(func, *args, **kwargs):
    start_time = time.time()
    result = func(*args, **kwargs)
    end_time = time.time()
    elapsed_time = end_time - start_time
    return result, elapsed_time, start_time, end_time
