"""
datas_preparation_RNN.py

Goal: turn the same labelled sensor CSV used for the SVM into a dataset
the RNN can be trained on, this time built around the trained SVMs'
predicted probabilities instead of raw features - and shifted forward in
time by DELTA_SAMPLES, so the RNN learns to predict the phase that will
occur at t + delta rather than the phase happening right now.

Overview of what happens below:
    1. window_layout_RNN() is run three times (once per SVM window size),
       sliding a window across the CSV and asking the corresponding
       trained SVM pipeline for its class probabilities at each window.
    2. crossing_prediction() averages the three SVMs' probabilities
       together, combining the "sees short phases well" window with the
       "more accurate features" windows into one probability per class.
    3. fusion_multi_windowing() then cuts that stream of averaged
       probabilities into overlapping sequences of length
       SEQUENCE_LENGTH, each one paired with the (shifted) label that
       should follow it - this is the actual (X, y) dataset the RNN
       will train on.
"""

import joblib
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import make_pipeline
import numpy as np
import glob

# -----------CONFIG--------------------------

DATAS = input("datas csv file name ?") + ".csv"

NUM_SENSORS = 3  # change this number for the number of sensors used

FIRST_WINDOW_SIZE = 13
SECOND_WINDOW_SIZE = 26
THIRD_WINDOW_SIZE = 37

# We can't compute a full-size window until we have at least
# max(window sizes) rows of history behind us, so the very first index
# we're allowed to look at is one row before that threshold.
START_INDEX = max(FIRST_WINDOW_SIZE, SECOND_WINDOW_SIZE, THIRD_WINDOW_SIZE) - 1

STEP = 1


FS_IMU = 74                                        # IMU sampling frequency, in Hz
DELAY_PROGRAMS_IN_SECONDS = 0.058                  # measured reaction delay of the full pipeline + motor
DELTA_SAMPLES = round(FS_IMU * DELAY_PROGRAMS_IN_SECONDS)  # that same delay, expressed in samples instead of seconds
SEQUENCE_LENGTH = DELTA_SAMPLES                    # length of the input sequence fed to the RNN

# --------------------------------------------


def crossing_prediction(first_prediction, second_prediction, third_prediction):
    """
    Combine the three SVMs' (one per window size) predicted class
    probabilities into a single probability array, by averaging them
    element-wise across the three window sizes.
    """
    predictions = np.stack((first_prediction, second_prediction, third_prediction))
    crossed_predictions = np.mean(predictions, axis=0)
    return (crossed_predictions)


def window_layout_RNN(window_size, step, csv_filepath, trained_pipeline):
    """
    Slide a window of `window_size` rows across the CSV, compute the same
    per-sensor features as datas_preparation_SVM.py, and ask
    `trained_pipeline` (an already-trained SVM pipeline for this exact
    window size) for its predicted class probabilities on each window.

    Unlike the SVM version, the label kept for each window isn't the
    label AT that window - it's the label DELTA_SAMPLES rows further
    ahead (see the `labels.shift(-DELTA_SAMPLES)` line below). This is
    what teaches the downstream RNN to anticipate the upcoming phase
    instead of just recognising the current one.

    Returns a list of shifted labels and the matching array of predicted
    probabilities, one entry per kept window.
    """

    csv_file = pd.read_csv(csv_filepath)
    labels = csv_file["labels"]

    # Shift the labels backward by DELTA_SAMPLES rows: the label now
    # stored at row `index` is really the label that will occur
    # DELTA_SAMPLES samples later. Rows near the end of the file won't
    # have a "future" label to shift in, so they become NaN.
    labels = labels.shift(-DELTA_SAMPLES)
    csv_datas = csv_file.drop(columns=['labels'], inplace=False)

    all_features = []
    shifted_label = []

    index = START_INDEX

    while index < len(csv_datas) - 1:
        if pd.isna(labels[index]):
            # No future label available for this row (too close to the
            # end of the recording, or landed in an unlabelled gap) -> skip.
            index += step
            continue
        else:
            features_sensors = {}

            # Same 5 features per sensor as in datas_preparation_SVM.py
            # (mean, std, min, max, growth rate), just computed on the
            # window ENDING at `index` instead of starting at it.
            for i in range(1, NUM_SENSORS + 1):
                roll_mean = float((csv_datas[f"roll_sensor{i}"][index - window_size + 1: index + 1]).mean())
                roll_std = float((csv_datas[f"roll_sensor{i}"][index - window_size + 1: index + 1]).std())
                roll_min = float(min(csv_datas[f"roll_sensor{i}"][index - window_size + 1: index + 1]))
                roll_max = float(max(csv_datas[f"roll_sensor{i}"][index - window_size + 1: index + 1]))
                roll_growth_rate = float((csv_datas[f"roll_sensor{i}"][index + 1] - csv_datas[f"roll_sensor{i}"][index - window_size + 1]) / window_size)

                pitch_mean = float((csv_datas[f"pitch_sensor{i}"][index - window_size + 1: index + 1]).mean())
                pitch_std = float((csv_datas[f"pitch_sensor{i}"][index - window_size + 1: index + 1]).std())
                pitch_min = float(min(csv_datas[f"pitch_sensor{i}"][index - window_size + 1: index + 1]))
                pitch_max = float(max(csv_datas[f"pitch_sensor{i}"][index - window_size + 1: index + 1]))
                pitch_growth_rate = float((csv_datas[f"pitch_sensor{i}"][index + 1] - csv_datas[f"pitch_sensor{i}"][index - window_size + 1]) / window_size)

                emg_mean = float((csv_datas[f"emg_rms_sensor{i}"][index - window_size + 1: index + 1]).mean())
                emg_std = float((csv_datas[f"emg_rms_sensor{i}"][index - window_size + 1: index + 1]).std())
                emg_max = float(max(csv_datas[f"emg_rms_sensor{i}"][index - window_size + 1: index + 1]))
                emg_growth_rate = float((csv_datas[f"emg_rms_sensor{i}"][index + 1] - csv_datas[f"emg_rms_sensor{i}"][index - window_size + 1]) / window_size)

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

                                          f"emg_growth_rate{i}": emg_growth_rate})
            # Note: the (already-shifted) label at `index` is kept
            # alongside these features, not fetched from `csv_file`
            # again - it's the label DELTA_SAMPLES rows in the future.
            shifted_label.append(labels[index])
            all_features.append(features_sensors)

            index += step

    all_features = pd.DataFrame(all_features)
    # Ask the already-trained SVM (for this specific window size) for its
    # predicted probability of each class, at every kept window.
    probabilities = trained_pipeline.predict_proba(all_features)

    return (shifted_label, probabilities)


def fusion_multi_windowing(first_window_size, second_window_size, third_window_size,
                            first_trained_pipeline, second_trained_pipeline, third_trained_pipeline,
                            csv_filepath, step, label_encoder):
    """
    Build the final RNN training set:
      1. Run window_layout_RNN() once per window size / trained SVM pipeline.
      2. Average the three SVMs' probabilities together (crossing_prediction).
      3. Cut the resulting probability stream into overlapping sequences
         of SEQUENCE_LENGTH probability-vectors, each paired with the
         label that should follow that sequence.
      4. Encode those labels to integers with the (already-fitted)
         label_encoder, and save both arrays to disk as .npy files.

    Returns (crossed_prediction_extracts, encoded_labels), the RNN's
    (X, y) training data.
    """
    (shifted_label_window_size1, probabilities_window_size1) = window_layout_RNN(first_window_size, step, csv_filepath, first_trained_pipeline)
    (shifted_label_window_size2, probabilities_window_size2) = window_layout_RNN(second_window_size, step, csv_filepath, second_trained_pipeline)
    (shifted_label_window_size3, probabilities_window_size3) = window_layout_RNN(third_window_size, step, csv_filepath, third_trained_pipeline)

    crossed_predictions = crossing_prediction(probabilities_window_size1, probabilities_window_size2, probabilities_window_size3)
    # The three window sizes are built from the same underlying rows and
    # the same shift, so their shifted labels line up 1-to-1; we only
    # need to keep one of the three lists.
    shifted_labels = shifted_label_window_size1
    shifted_label_extracts = []
    crossed_prediction_extracts = []
    index = SEQUENCE_LENGTH

    # Slide a window of length SEQUENCE_LENGTH over the averaged
    # probabilities: each extract is one sequence of past probability
    # vectors, paired with the (already future-shifted) label right
    # after that sequence - i.e. "given this recent history of SVM
    # probabilities, what phase comes next?"
    while index < len(shifted_labels) - 1:
        shifted_label_extracts.append(shifted_labels[index])
        crossed_prediction_extracts.append(crossed_predictions[index - SEQUENCE_LENGTH: index])
        index += 1

    encoded_labels = label_encoder.transform(shifted_label_extracts)
    shifted_label_extracts = np.array(shifted_label_extracts)
    crossed_prediction_extracts = np.array(crossed_prediction_extracts)
    np.save("encoded_labels", encoded_labels)
    np.save("crossed_prediction_extracts", crossed_prediction_extracts)
    return (crossed_prediction_extracts, encoded_labels)


if __name__ == "__main__":
    # Reload the three SVM pipelines trained by SVM.py (one per window
    # size) and the label encoder fitted by encoding_labels(), then build
    # the RNN's training data from them.
    SVM_results = {"trained_pipeline_window13": joblib.load(f"pipe_features_windowsize_of_13_session20260612_143920 - Copie.pkl"),
                   "trained_pipeline_window26": joblib.load(f"pipe_features_windowsize_of_26_session20260612_143920 - Copie.pkl"),
                   "trained_pipeline_window37": joblib.load(f"pipe_features_windowsize_of_37_session20260612_143920 - Copie.pkl")}
    label_encoder = joblib.load(glob.glob("label_encoder*.pkl")[0])
    (crossed_prediction_extracts, encoded_labels) = fusion_multi_windowing(FIRST_WINDOW_SIZE, SECOND_WINDOW_SIZE, THIRD_WINDOW_SIZE,
                                                                            SVM_results["trained_pipeline_window13"], SVM_results["trained_pipeline_window26"], SVM_results["trained_pipeline_window37"],
                                                                            DATAS, STEP, label_encoder)