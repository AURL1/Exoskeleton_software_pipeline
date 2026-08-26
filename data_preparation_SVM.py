"""
datas_preparation_SVM.py

Goal: turn the labelled, timestamped sensor CSV (produced by
json_and_csv_fusion.py) into a features CSV that the SVM can be trained on.

Instead of feeding the SVM raw sensor values row by row, we slide a window
over the data and compute summary statistics (mean, std, min, max, growth
rate) over each window. Each window becomes one row of the features table,
labelled with whichever "labels" value the window falls under.

This is run three times, once per window size (see multi_windowing below),
because:
    - a window that's too LONG risks straddling two different labelled
      phases, or skipping short phases entirely.
    - a window that's too SHORT gives a noisier estimate of each feature
      (fewer samples to average over).
Combining several window sizes downstream is meant to get the best of
both: good coverage of every labelled phase, and reasonably accurate
features.
"""

import pandas as pd
import numpy as np

# -----------------CONFIG---------------------
DATAS = input("datas csv file name ?") + ".csv"

FIRST_WINDOW_SIZE = 37
SECOND_WINDOW_SIZE = 26
THIRD_WINDOW_SIZE = 13
SENSORS_NUMBER = 3

# --------------------------------------------


def window_layout_SVM(window_size, step, csv_filepath):
    """
    Slide a window of `window_size` rows (moving forward by `step` rows
    each time) across the CSV at `csv_filepath`, and compute one row of
    features per window. Windows that contain more than one distinct
    label, or that fall in an unlabelled ('') region, are skipped.

    Saves the resulting features table to
    "features_windowsize_of_{window_size}_{csv_filepath}" and returns it
    as a DataFrame.
    """

    csv_datas = pd.read_csv(csv_filepath, keep_default_na=False)

    all_features = []

    index = 0

    while index < (len(csv_datas) - window_size - 1):

        # Only keep this window if every row inside it shares the exact
        # same label (set(...) == 1 unique value means "no mix of labels").
        if len(set(csv_datas['labels'][index:index + window_size])) == 1:

            # An empty label means this window falls in an unlabelled
            # region of the recording (e.g. between two exercises) -> skip it.
            if csv_datas['labels'][index] == '':
                index += step
                continue
            else:
                features_sensors = {}

                # Compute the same 5 features (mean, std, min, max, growth
                # rate) for roll, pitch and EMG, for every sensor.
                for i in range(1, SENSORS_NUMBER + 1):
                    roll_mean = float((csv_datas[f"roll_sensor{i}"][index:index + window_size]).mean())
                    roll_std = float((csv_datas[f"roll_sensor{i}"][index:index + window_size]).std())
                    roll_min = float(min(csv_datas[f"roll_sensor{i}"][index:index + window_size]))
                    roll_max = float(max(csv_datas[f"roll_sensor{i}"][index:index + window_size]))
                    # "growth rate" here = (last value - first value) / window_size,
                    # i.e. average rate of change of the signal across the window.
                    roll_growth_rate = float((csv_datas[f"roll_sensor{i}"][index + window_size - 1] - csv_datas[f"roll_sensor{i}"][index]) / window_size)

                    pitch_mean = float((csv_datas[f"pitch_sensor{i}"][index:index + window_size]).mean())
                    pitch_std = float((csv_datas[f"pitch_sensor{i}"][index:index + window_size]).std())
                    pitch_min = float(min(csv_datas[f"pitch_sensor{i}"][index:index + window_size]))
                    pitch_max = float(max(csv_datas[f"pitch_sensor{i}"][index:index + window_size]))
                    pitch_growth_rate = float((csv_datas[f"pitch_sensor{i}"][index + window_size - 1] - csv_datas[f"pitch_sensor{i}"][index]) / window_size)

                    emg_mean = float((csv_datas[f"emg_rms_sensor{i}"][index:index + window_size]).mean())
                    emg_std = float((csv_datas[f"emg_rms_sensor{i}"][index:index + window_size]).std())
                    emg_max = float(max(csv_datas[f"emg_rms_sensor{i}"][index:index + window_size]))
                    # emg_squared_sum = float(sum(np.square(csv_datas[f"emg_rms_sensor{i}"][index:index+window_size]))/window_size)
                    emg_growth_rate = float((csv_datas[f"emg_rms_sensor{i}"][index + window_size - 1] - csv_datas[f"emg_rms_sensor{i}"][index]) / window_size)

                    # All windows in this loop share the same label, so we
                    # can just read it once from the first row of the window.
                    label = csv_datas["labels"][index]

                    # Store all features for sensor i under keys suffixed
                    # with the sensor number (roll_mean1, roll_mean2, ...),
                    # so every sensor's features end up as separate columns.
                    features_sensors.update({f"roll_mean{i}": roll_mean,
                                              f"roll_std{i}": roll_std,
                                              f"roll_min{i}": roll_min,
                                              f"roll_max{i}": roll_max,
                                              f"roll_growth_rate{i}": roll_growth_rate,

                                              f"pitch_mean{i}": pitch_mean,
                                              f"pitch_std{i}": pitch_std,
                                              f"pitch_min{i}": pitch_min,
                                              f"pitch_max{i}": pitch_max,
                                              f"pitch_growth_rate{i}": pitch_growth_rate,

                                              f"emg_mean{i}": emg_mean,
                                              f"emg_std{i}": emg_std,
                                              f"emg_max{i}": emg_max,
                                              #  f"emg_squared_sum{i}":emg_squared_sum,
                                              f"emg_growth_rate{i}": emg_growth_rate,

                                              "label": label})

                # One dict of features (all sensors combined) = one row
                # of the future features table.
                all_features.append(features_sensors)

                index += step

        else:
            # Mixed-label window (straddles a label boundary) -> skip it,
            # slide forward, and try again.
            index += step
            continue

    all_features = pd.DataFrame(all_features)
    print(f"all features (DataFrame):\n{all_features}")
    all_features.to_csv(f"features_windowsize_of_{window_size}_{csv_filepath}", index=False)

    return (all_features)


def multi_windowing(window_size1, window_size2, window_size3):
    """
    Run window_layout_SVM three times, once per window size, each time
    with a step of half the window size (50% overlap between windows).
    Produces three separate features CSVs, one per window size.
    """
    window_layout_SVM(window_size1, step=int(window_size1 / 2), csv_filepath=DATAS)
    window_layout_SVM(window_size2, step=int(window_size2 / 2), csv_filepath=DATAS)
    window_layout_SVM(window_size3, step=int(window_size3 / 2), csv_filepath=DATAS)


if __name__ == "__main__":
    multi_windowing(FIRST_WINDOW_SIZE, SECOND_WINDOW_SIZE, THIRD_WINDOW_SIZE)