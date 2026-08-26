"""
json_and_csv_fusion.py

Goal: attach a movement-phase label to every row of a sensor CSV file,
using the label intervals (start/end timestamps) exported from Label Studio
as a JSON file.

Input:
    - a JSON file exported from Label Studio, containing a list of
      labelled time intervals (start, end, timeserieslabels)
    - a CSV file containing the resampled IMU/EMG data, with a "time_s" column

Output:
    - a new CSV file ("labellised_<original_name>.csv") identical to the
      input CSV, but with an extra "labels" column filled in.
"""

import json
import pandas as pd
import numpy as np
import os

# -----------------CONFIG---------------------

LABELS = input("Name of the json file ?") + ".json"
DATAS = input("Name of the csv file ?") + ".csv"

# --------------------------------------------


def json_and_csv_fusion(json_filepath, csv_filepath):
    """
    Merge a CSV of timestamped sensor data with a JSON file of labelled
    time intervals (exported from Label Studio), producing a new CSV
    where each row is tagged with the label whose interval contains it.
    """

    try:
        csv_datas = pd.read_csv(csv_filepath)

        with open(json_filepath, 'r') as json_file:
            opened_json_file = json.load(json_file)

            # Label Studio does not guarantee the labels are exported in
            # chronological order, so we sort them by their start time first.
            ordered_json_labels = sorted(
                opened_json_file[0]['label'],
                key=lambda item: item['start']
            )

            # Add an empty "labels" column at the end of the CSV, to be
            # filled in below.
            last_column = len(csv_datas.columns)
            csv_datas.insert(last_column, "labels", "")

            # ----------------------------------NAIVE ALGORITHM----------------------------------
            # This first version compared every CSV row against every label
            # interval with two nested loops (O(n_labels * n_rows)).
            # It works, but is slow on large recordings, hence the
            # vectorised version used below.
            #
            # for i in range(len(ordered_json_labels)):
            #   for j in csv_datas.index:
            #     if (csv_datas["time_s"][j]>ordered_json_labels[i]['start']) & (csv_datas["time_s"][j]<ordered_json_labels[i]['end']):
            #         csv_datas.loc[j,"labels"]=ordered_json_labels[i]['timeserieslabels']

            # -------------OPTIMISED ALGORITHM USING np.array AND USING INTERVALS---------------

            # Step 1: split the JSON label entries into three parallel lists:
            # interval start times, interval end times, and the label name
            # attached to each interval.
            start = []
            end = []
            sublists_labels = []
            labels = []

            for i in range(len(ordered_json_labels)):
                start.append(ordered_json_labels[i]['start'])
                end.append(ordered_json_labels[i]['end'])
                sublists_labels.append(ordered_json_labels[i]['timeserieslabels'])

            # Label Studio stores each label as a one-item list
            # (e.g. ["walking"]), so we unwrap it to get a flat list of strings.
            for i in range(len(sublists_labels)):
                labels.append(sublists_labels[i][0])

            labels_array = np.array(labels)

            # Step 2: build a pandas IntervalIndex from the (start, end) pairs.
            # closed="neither" means the interval excludes both of its
            # endpoints (start < time_s < end), matching the naive algorithm above.
            slices = pd.IntervalIndex.from_arrays(start, end, closed="neither")

            # Step 3: for every timestamp in the CSV, find which interval
            # (if any) it falls into. get_indexer returns, for each
            # timestamp, the position of the matching interval in `slices`,
            # or -1 if the timestamp isn't inside any interval.
            index = pd.Index.get_indexer(slices, target=csv_datas['time_s'])

            # Step 4: use those positions to pull the corresponding label
            # for every row in one vectorised operation (no loop needed).
            csv_datas["labels"] = labels_array[index]

            # Step 5: rows that didn't fall inside any labelled interval
            # got assigned the last element of labels_array by mistake
            # (because Python allows negative indexing). We fix that here
            # by blanking out any row where get_indexer returned -1.
            mask = (index == -1)
            csv_datas.loc[mask, "labels"] = ''

            # ----------------------------------------------------------------------------------

            # Save the result as "labellised_<original csv name>.csv"
            csv_name = os.path.splitext(csv_filepath)[0]
            csv_datas.to_csv(f"labellised_{csv_name}.csv", index=False)

    except FileNotFoundError:
        print(f"Error : One of the file was not found.")
        return


if __name__ == "__main__":
    json_and_csv_fusion(LABELS, DATAS)