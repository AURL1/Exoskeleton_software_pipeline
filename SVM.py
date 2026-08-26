"""
SVM.py

Goal: train an SVM classifier on the features CSVs produced by
datas_preparation_SVM.py (one classifier per window size), evaluate it
with time-aware cross-validation, and save both the trained model and
the label encoder to disk so they can be reused later
(by datas_preparation_RNN.py and realtime_inference_pipeline.py).
"""

from sklearn import svm
from sklearn.model_selection import train_test_split, TimeSeriesSplit, cross_validate, cross_val_predict
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import make_pipeline
from sklearn.metrics import classification_report
import pandas as pd
import numpy as np
import joblib
import os

# -----------------CONFIG---------------------

DATAS_1 = input("First datas csv file name ?") + ".csv"
DATAS_2 = input("Second datas csv file name ?") + ".csv"
DATAS_3 = input("Third datas csv file name ?") + ".csv"

# --------------------------------------------


def encoding_labels(csv_filepath):
    """
    Fit a LabelEncoder on the string labels of the given CSV (turning
    e.g. "walking"/"standing" into 0/1/2...), save the fitted encoder to
    disk as a .pkl file, and return it.

    Note: only the encoder object is saved/returned here - the encoded
    values themselves aren't used elsewhere in this script (the SVM below
    is trained directly on the string labels, which scikit-learn's SVC
    handles natively). This encoder is meant to be reloaded later, e.g.
    to decode the RNN or realtime pipeline's numeric predictions back
    into human-readable label names.
    """
    csv_file = pd.read_csv(csv_filepath)
    labels = csv_file["label"]
    label_encoder = LabelEncoder()
    label_encoder.fit_transform(labels)
    filepath_name_with_extension = os.path.basename(csv_filepath)
    filepath_name = os.path.splitext(filepath_name_with_extension)[0]
    joblib.dump(label_encoder, f"label_encoder_{filepath_name}.pkl")
    return (label_encoder)


def build_pipeline(X, y):
    """
    Build and fit a scikit-learn pipeline that first standardises the
    features (zero mean, unit variance) and then feeds them to an SVM
    with an RBF kernel. class_weight='balanced' compensates for phases
    that appear less often than others in the training data.
    """
    scaler = StandardScaler()
    svc = svm.SVC(kernel='rbf', probability=True, class_weight='balanced')
    pipeline = make_pipeline(scaler, svc)
    pipeline.fit(X, y)
    return (pipeline)


def SVM(csv_filepath):
    """
    Train and evaluate an SVM on one features CSV (one window size).

    Evaluation uses TimeSeriesSplit instead of a regular train/test split
    or k-fold: because the data is a time series, a normal random split
    would let the model train on "future" windows and test on "past"
    ones, which leaks information. TimeSeriesSplit instead trains on an
    early chunk and always tests on the chunk that comes right after it,
    5 times over, sliding forward through the recording.

    After evaluation, a final pipeline is retrained on ALL the data
    (not just one fold) and saved to disk - that's the one meant to be
    used for real predictions afterwards.
    """

    csv_file = pd.read_csv(csv_filepath)
    labels = csv_file["label"]
    csv_datas = csv_file.drop(columns=['label'], inplace=False)
    splitter = TimeSeriesSplit(5)
    y_pred = np.full(labels.shape[0], None)
    y_pred_proba = np.full((labels.shape[0], len(labels.unique())), None)

    # TimeSeriesSplit doesn't cover every single row (the very first fold's
    # training chunk has nothing before it to test on), so y_pred and
    # y_pred_proba start out full of `None` and only get filled in for the
    # rows that actually ended up in a test fold.
    for train_index, test_index in splitter.split(csv_datas):
        X_train = csv_datas.iloc[train_index]
        y_train = labels.iloc[train_index]
        X_test = csv_datas.iloc[test_index]

        fold_pipeline = build_pipeline(X_train, y_train)
        fold_prediction = fold_pipeline.predict(X_test)
        y_pred[test_index] = fold_prediction

        fold_prediction_proba = fold_pipeline.predict_proba(X_test)
        y_pred_proba[test_index] = fold_prediction_proba

    # Keep only the rows that were actually predicted (i.e. drop the
    # leftover `None` rows from before the first fold) before scoring.
    mask = (y_pred != None)
    masked_labels = labels[mask]
    masked_y_pred = y_pred[mask]
    print(f"{classification_report(masked_labels, masked_y_pred)}")

    # Same masking logic, but for the predicted probabilities: mask_proba
    # is True for any row where at least one class probability was filled in.
    mask_proba_bool = (y_pred_proba != None)
    mask_proba = mask_proba_bool.any(axis=1)
    masked_labels_proba = labels[mask_proba]
    masked_y_pred_proba = y_pred_proba[mask_proba]

    # The cross-validation above is only there to measure performance.
    # The model that actually gets saved and reused is retrained on the
    # full dataset, so it benefits from every available sample.
    global_pipeline = build_pipeline(csv_datas, labels)

    filepath_name_with_extension = os.path.basename(csv_filepath)
    filepath_name = os.path.splitext(filepath_name_with_extension)[0]
    joblib.dump(global_pipeline, f"pipe_{filepath_name}.pkl")

    return (global_pipeline, masked_labels_proba, masked_y_pred_proba)


def loop(csv_filepath1, csv_filepath2, csv_filepath3):
    """
    Run the full SVM() training + evaluation routine once per window-size
    features CSV, returning the three results as a tuple.
    """
    results_SVM1 = SVM(csv_filepath1)
    results_SVM2 = SVM(csv_filepath2)
    results_SVM3 = SVM(csv_filepath3)
    return (results_SVM1, results_SVM2, results_SVM3)


if __name__ == "__main__":
    encoding_labels(DATAS_1)
    loop(DATAS_1, DATAS_2, DATAS_3)