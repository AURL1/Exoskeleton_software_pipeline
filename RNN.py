"""
RNN.py

Goal: train the LSTM that predicts the upcoming gait phase (at t+delta)
from the sequences of averaged SVM probabilities produced by
datas_preparation_RNN.py.

Pipeline in this file:
    1. Split the (X, y) data saved by datas_preparation_RNN.py into
       train / validation / test sets, in that chronological order
       (no shuffling - see note in RNN() below).
    2. Train the LSTM with early stopping: after every epoch, if the
       validation loss improved, save the model; if it didn't improve
       for PATIENCE epochs in a row, stop training early.
    3. Reload the best saved model and evaluate it on the held-out test
       set, printing a classification report and confusion matrix.
    4. Save that final model to disk under a timestamped filename.
"""

from sklearn import model_selection
import glob
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix
from datetime import datetime

# -------------------------CONFIG-------------------------

DATASET_TRAIN_PROPORTION = 0.7
DATASET_VALIDATION_PROPORTION = 0.15
# (the remaining 1 - 0.7 - 0.15 = 0.15 is used for the test set)
EPOCH_NUMBER = 150
PATIENCE = 10  # stop training if validation loss hasn't improved for this many epochs in a row

# ----------------------------------------------------------


class LSTM(nn.Module):
    """
    A single-layer LSTM followed by dropout and a linear layer, mapping
    a sequence of SVM probability vectors (input_size=4, i.e. one value
    per gait-phase class) to logits over the same 4 classes.

    Only the LAST hidden state of the sequence (h_n[0]) is used for
    classification - i.e. the model reads the whole input sequence, then
    makes one prediction based on what it "remembers" at the end of it.
    """
    def __init__(self):
        super().__init__()
        self.LSTM = nn.LSTM(input_size=4, hidden_size=16, num_layers=1, batch_first=True)
        self.Dropout = nn.Dropout(0.2)
        self.Linear = nn.Linear(in_features=16, out_features=4)

    def forward(self, x):
        (output, (h_n, c_n)) = self.LSTM(x)
        hidden = h_n[0]
        dropped = self.Dropout(hidden)
        logits = self.Linear(dropped)
        return (logits)


def RNN(crossed_prediction_extracts, shifted_label_extracts):
    """
    Train and evaluate the LSTM defined above.

    crossed_prediction_extracts: array of input sequences (X), as saved
        by datas_preparation_RNN.py.
    shifted_label_extracts: array of matching future-shifted labels (y),
        already integer-encoded.
    """

    # --- Train / validation / test split ---
    # Note: this is a plain chronological slice (first 70% -> train,
    # next 15% -> validation, last 15% -> test), NOT a random split.
    # Since this is time-series data, shuffling would let the model
    # train on windows that come after the ones used to validate/test
    # it, which would leak information from the "future".
    label_train_set = shifted_label_extracts[0:int(DATASET_TRAIN_PROPORTION * len(shifted_label_extracts))]
    label_validation_set = shifted_label_extracts[int(DATASET_TRAIN_PROPORTION * len(shifted_label_extracts)):int((DATASET_TRAIN_PROPORTION + DATASET_VALIDATION_PROPORTION) * len(shifted_label_extracts))]
    label_test_set = shifted_label_extracts[int((DATASET_TRAIN_PROPORTION + DATASET_VALIDATION_PROPORTION) * len(shifted_label_extracts)):len(shifted_label_extracts)]

    label_train_tensor = torch.from_numpy(label_train_set)
    label_validation_tensor = torch.from_numpy(label_validation_set)
    label_test_tensor = torch.from_numpy(label_test_set)

    # Same chronological split, applied to the input sequences this time.
    training_data = crossed_prediction_extracts[0:int(DATASET_TRAIN_PROPORTION * len(crossed_prediction_extracts))]
    validation_data = crossed_prediction_extracts[int(DATASET_TRAIN_PROPORTION * len(crossed_prediction_extracts)):int((DATASET_TRAIN_PROPORTION + DATASET_VALIDATION_PROPORTION) * len(crossed_prediction_extracts))]
    test_data = crossed_prediction_extracts[int((DATASET_TRAIN_PROPORTION + DATASET_VALIDATION_PROPORTION) * len(crossed_prediction_extracts)):len(crossed_prediction_extracts)]

    # PyTorch tensors need float32, not the float64 numpy defaults to.
    training_data = np.float32(training_data)
    validation_data = np.float32(validation_data)
    test_data = np.float32(test_data)

    data_train_tensor = torch.from_numpy(training_data)
    dataset_validation_tensor = torch.from_numpy(validation_data)
    dataset_test_tensor = torch.from_numpy(test_data)

    # Pair each set of inputs with its matching labels.
    train_dataset = TensorDataset(data_train_tensor, label_train_tensor)
    validation_dataset = TensorDataset(dataset_validation_tensor, label_validation_tensor)
    test_dataset = TensorDataset(dataset_test_tensor, label_test_tensor)

    # DataLoaders hand out the data in mini-batches of 256 samples
    # instead of feeding the whole dataset to the model at once.
    train_batchs = DataLoader(train_dataset, batch_size=256)
    validation_batchs = DataLoader(validation_dataset, batch_size=256)
    test_batchs = DataLoader(test_dataset, batch_size=256)

    # Some gait phases likely appear less often than others in the
    # training data (e.g. brief transition phases vs. long steady
    # walking); class_weight='balanced' computes a weight per class so
    # the loss penalises mistakes on rare classes more, preventing the
    # model from just always predicting the most common phase.
    balance = torch.from_numpy(np.float32(compute_class_weight('balanced', classes=np.unique(label_train_set), y=label_train_set)))
    criterion = nn.CrossEntropyLoss(weight=balance)
    train_and_validation_lstm = LSTM()
    optimizer = torch.optim.Adam(train_and_validation_lstm.parameters(), lr=0.001)
    best_val_loss = float('inf')
    epochs_without_improvement = 0

    # --- Training loop with early stopping ---
    for epoch in range(EPOCH_NUMBER):
        loss_batchs_train = []
        loss_batchs_validation = []

        # model.train() switches on dropout for the training pass.
        train_and_validation_lstm.train()
        for X_batch, y_batch in train_batchs:
            optimizer.zero_grad()                       # reset gradients from the previous batch
            logits = train_and_validation_lstm(X_batch)  # forward pass
            loss_train = criterion(logits, y_batch)
            loss_batchs_train.append(loss_train.item())
            loss_train.backward()                        # backward pass: compute gradients
            optimizer.step()                              # update the model's weights

        # model.eval() + torch.no_grad() switches dropout off and skips
        # gradient tracking, since we're only measuring performance here,
        # not training on the validation set.
        train_and_validation_lstm.eval()
        with torch.no_grad():
            for X_batch, y_batch in validation_batchs:
                logits = train_and_validation_lstm(X_batch)
                loss_validation = criterion(logits, y_batch)
                loss_batchs_validation.append(loss_validation.item())

        epoch_loss_train = sum(loss_batchs_train) / len(loss_batchs_train)
        epoch_loss_validation = sum(loss_batchs_validation) / len(loss_batchs_validation)

        # Early stopping: keep the model from the epoch with the lowest
        # validation loss seen so far. If several epochs in a row fail
        # to beat that best score, stop training - it's a sign the model
        # has started overfitting rather than genuinely improving.
        if best_val_loss > epoch_loss_validation:
            epochs_without_improvement = 0
            best_val_loss = epoch_loss_validation
            torch.save(train_and_validation_lstm.state_dict(), "best_LSTM_model.pt")
        else:
            epochs_without_improvement += 1

        print(f"epoch_loss_train at epoch {epoch} : {epoch_loss_train}")
        print(f"epoch_loss_validation at epoch {epoch} : {epoch_loss_validation}")

        if epochs_without_improvement == PATIENCE:
            break

    # --- Final evaluation on the held-out test set ---
    predicted_label_phases = []
    true_labels_test = []
    test_lstm = LSTM()
    # Reload the BEST checkpoint saved during training (not necessarily
    # the one from the very last epoch), since early stopping may have
    # triggered several epochs after the actual best one.
    best_lstm_model = torch.load("best_LSTM_model.pt")
    test_lstm.load_state_dict(best_lstm_model)
    with torch.no_grad():
        test_lstm.eval()
        for X_batch, y_batch in test_batchs:
            logits = test_lstm(X_batch)
            prediction = torch.argmax(logits, dim=1)  # pick the class with the highest logit
            prediction = prediction.numpy()
            y_batch = y_batch.numpy()
            predicted_label_phases.append(prediction)
            true_labels_test.append(y_batch)
        predicted_label_phases = np.concatenate(predicted_label_phases)
        true_labels_test = np.concatenate(true_labels_test)
        print(f"{classification_report(true_labels_test, predicted_label_phases)}")
        print(f"{confusion_matrix(true_labels_test, predicted_label_phases)}")

        # Save the final evaluated model under a timestamped filename, so
        # it doesn't get overwritten by the next training session's
        # "best_LSTM_model.pt" checkpoint.
        session_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        torch.save(test_lstm.state_dict(), f"phase_prediction_lstm{session_str}.pt")


if __name__ == "__main__":
    # Reload the (X, y) arrays saved by datas_preparation_RNN.py and
    # train the LSTM on them.
    shifted_label_extracts = np.load(glob.glob("encoded_labels*.npy")[0])
    crossed_prediction_extracts = np.load(glob.glob("crossed_prediction_extracts*.npy")[0])
    RNN(crossed_prediction_extracts, shifted_label_extracts)