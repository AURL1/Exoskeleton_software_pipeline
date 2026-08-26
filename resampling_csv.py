"""
This script resamples a CSV file of sensor measurements (produced by
angles_and_EMG_measurements.py) onto a single, regular time base, so that
every column (IMU angles and EMG) is aligned at the same sampling frequency
before labelling.

Previous script: angles_and_EMG_measurements.py
Next script: json_and_csv_fusion.py

Usage:
When run directly, the script asks for the base name of a session CSV file
(the same name printed by angles_and_EMG_measurements.py, without the
".csv" extension), then writes a new file named "<name>_resampled.csv" in
the same folder.

What it does, step by step:
-  Load the CSV and convert its "time_s" column (seconds since the
   recording started) into real datetime values, anchored to the current
   time (t0).
- Use that datetime column as the index, so pandas' time-based resampling
   tools can be used.
- Resample onto a fixed time grid (RESAMPLE_INTERVAL_MS) and fill the new
   timestamps by linear interpolation in time.
- Convert the datetime index back into a plain "seconds since t0" column,
   matching the original format.
- Save the result to "<name>_resampled.csv".
"""

import pandas as pd
from datetime import datetime

#-----------CONFIG--------------------------

# Target sampling period after resampling, in milliseconds.
# It corresponds to ~74 Hz, matching FS_IMU in
# angles_and_EMG_measurements.py, so that the resampled IMU and EMG columns
# line up with the original IMU sampling rate.
RESAMPLE_INTERVAL_MS = 13.5

SENSOR_MEASUREMENTS_FILE_NAME = input("Sensor measurements file name ?")
SENSOR_MEASUREMENTS_FILE_CSV = SENSOR_MEASUREMENTS_FILE_NAME + ".csv"

#-------------------------------------------

def resampling_csv(file_name, file_csv_filepath):

    try:
        t0 = datetime.now()

        datas = pd.read_csv(file_csv_filepath)

        # Turn "time_s" (seconds) into real datetime values
        time_s = datas['time_s']
        time_s = pd.to_datetime(time_s, unit='s', origin=t0)
        datas['time_s'] = time_s

        # Use the datetime column as index for resampling
        datas = datas.set_index("time_s")

        # Resample onto a fixed time grid + interpolate
        datas = datas.resample(pd.Timedelta(milliseconds=RESAMPLE_INTERVAL_MS)).asfreq().interpolate(method="time")

        # Convert the datetime index back to seconds since t0
        datas = datas.reset_index()
        datas['time_s'] = (datas['time_s'] - t0).dt.total_seconds()

        # Save the resampled data
        datas.to_csv(f"{file_name}_resampled.csv")

    except FileNotFoundError:
        print(f"Error : The file '{file_csv_filepath}' was not found.")
        return

if __name__ == '__main__':
    resampling_csv(SENSOR_MEASUREMENTS_FILE_NAME, SENSOR_MEASUREMENTS_FILE_CSV)