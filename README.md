# Design and Control of a Lower-Limb Exoskeleton

Hip exoskeleton built from Delsys Trigno Avanti sensors (IMU + EMG) and a CubeMars AK60-6 V3.0 KV80 actuator, whose motor assistance is driven not by the wearer's current position but by a **prediction of the gait phase that will occur shortly after** — anticipating the movement instead of reacting to it.

This project was carried out as a 2025–2026 internship at the **Embodied AI and Neurorobotics Lab**, Maersk Mc-Kinney Moller Institute (MMMI), University of Southern Denmark, under Polytech Nice — Department of Robotics. The full methodology, equations and results are detailed in the [internship report](#report) included in this repo.

## How it works

Two AI models are chained together:

1. **SVM** (Support Vector Machine) — recognises the *current* gait phase from IMU/EMG features.
2. **LSTM** (Long Short-Term Memory network) — fed the SVM's output, anticipates the phase that will occur after a delay δ (measured empirically on the assembled pipeline: ~58 ms).

The gait cycle is split into 4 phases: *double-support landing*, *single-leg support*, *dual-thrust*, *swing*. The motor only applies assistive torque during the two propulsive phases (dual-thrust, swing).

Two loops make up the project:

- **Offline loop** — run once, before wearing the suit, to record training data and train the SVM + LSTM.
- **Online loop** — run live, while wearing the suit, using the trained models to drive the motor in real time.

## Hardware

| Component | Role |
|---|---|
| 3× Delsys Trigno Avanti sensors | IMU (orientation, mode 39/quaternions) + EMG per sensor |
| CubeMars AK60-6 V3.0 KV80 | Hip actuator, driven over CAN bus in MIT mode |
| CAN–USB adapter (`slcan`) | Motor communication |
| 3D-printed PETG frame + aluminium profile + adhesive straps | Mechanical structure, torque transmission, body fixation |

## Repository structure

### Offline loop (train once, before wearing the suit)

| Script | Input → Output | Purpose |
|---|---|---|
| `pytrigno.py` | — | Vendor SDK wrapper (Delsys), extended with hand-derived quaternion → Euler angle calculations (not provided by Delsys) |
| `angles_and_EMG_measurements.py` | Trigno sensors (live) → `sessionYYYYMMDD_HHMMSS.csv` | Acquires IMU (roll/pitch) + EMG, filters both (band-pass + notch + RMS for EMG, low-pass for IMU), plots live, saves to CSV |
| `resampling_csv.py` | `session*.csv` → `*_resampled.csv` | Resamples every signal onto one uniform ~74 Hz clock (removes jitter from real-time acquisition) |
| *Label Studio (Docker)* | `*_resampled.csv` → `*.json` | Manual labelling of the 4 gait phases on the recorded session |
| `json_and_csv_fusion.py` | `*_resampled.csv` + `*.json` → `labellised_*.csv` | Attaches the Label Studio labels to each data row |
| `data_preparation_SVM.py` | `labellised_*.csv` → `features_windowsize_of_*.csv` | Extracts features (mean, std, min, max, growth rate) with 3 window sizes (13/26/37 rows) to balance short-phase coverage vs. feature stability |
| `SVM.py` | `features_windowsize_of_*.csv` → `pipe_*.pkl`, `label_encoder_*.pkl` | Trains one RBF-kernel SVC per window size (`class_weight="balanced"`), evaluated with `TimeSeriesSplit` to avoid data leakage |
| `data_preparation_RNN.py` | `labellised_*.csv` + trained SVM pipelines → `crossed_prediction_extracts.npy`, `encoded_labels.npy` | Runs the 3 SVMs, averages their probabilities, builds label-shifted sequences (t + δ) for the LSTM |
| `RNN.py` | `.npy` files → `best_LSTM_model.pt`, `phase_prediction_lstm*.pt` | Trains the LSTM (single layer, hidden size 16, dropout 0.2) with early stopping; prints classification report + confusion matrix |
| `latency_analysis.py` | `timestamps.csv` (from the online loop) → plots + δ estimate | Measures per-stage pipeline latency (windowing / SVM / LSTM) to set δ empirically |

### Online loop (real time, worn)

| Script | Purpose |
|---|---|
| `realtime_inference_pipeline.py` | Multithreaded real-time pipeline: acquires + filters IMU/EMG live, runs the 3 SVMs + LSTM every ~14 ms, applies a finite-state machine (confidence threshold, standing-still heuristic, safety watchdog) and sends torque commands to the CubeMars motor over CAN |

## Requirements

- Python 3.9+
- `numpy`, `pandas`, `scipy`, `scikit-learn`, `joblib`, `torch`, `matplotlib`, `python-can`
- Delsys Trigno Control Utility (TCU) running locally, sensors paired (see `pytrigno.py` and the Delsys SDK docs linked in `Trigno SDK and API links.txt`)
- Docker, to run a local Label Studio instance for labelling
- A CAN adapter supporting the `slcan` interface, for the CubeMars motor

```bash
pip install numpy pandas scipy scikit-learn joblib torch matplotlib python-can
```

## Usage

Run the offline loop once, in order, to produce the trained SVM pipelines and LSTM checkpoint, then run the online loop:

```
angles_and_EMG_measurements.py
        ↓
resampling_csv.py
        ↓
Label Studio (Docker) — manual labelling
        ↓
json_and_csv_fusion.py
        ↓
data_preparation_SVM.py  →  SVM.py
        ↓
data_preparation_RNN.py  →  RNN.py
        ↓
latency_analysis.py (sets δ, from a first realtime_inference_pipeline.py run)
        ↓
realtime_inference_pipeline.py  (worn, real time)
```

Each script prompts interactively for the filenames of the CSV/JSON files it needs.

## Results

- **SVM**: satisfactory F1 scores on most of the 4 phases; the shorter window size best captures the brief *dual-thrust* phase, larger windows give steadier features on the others — hence averaging the three.
- **LSTM**: 0.79 overall accuracy on the held-out test set. The *dual-thrust* phase is recognised best (F1 0.92) thanks to class balancing; remaining errors concentrate between temporally adjacent phases, consistent with the continuous, non-abrupt nature of gait transitions.

## Limitations & next steps

- The standing-still detector is still a hand-tuned heuristic, not learned.
- Trained on a single volunteer — extending to several people would improve robustness to inter-individual gait variability.
- Field-testing with the motor assistance actually engaged during walking is the natural next validation step (current results are on pre-recorded data).

## Report

<a name="report"></a>
The full internship report — *"Design and control of a lower limb exoskeleton"* — covers the mechanical design, the hand-derived quaternion calculations, the filtering choices, and a full glossary (FR/EN) of the technical terms used in this project. See `main.tex` / the PDF export in this repo.

## Author

**Aurélien Dorolle** — 4th year student, Polytech Nice, Department of Robotics
Internship host: Xiaofeng Xiong (SDU, tutor), Cao Danh Do (SDU), Roula Nassif (Polytech Nice, referee professor)

## References

- CubeMars, *AK Series Module Product Manual*, v3.2.0, 2024.
- Queen's Biomechatronics Team (QBMET), *Application of CubeMars AK Series Robotic Actuators for a Lower Body Assistive Exoskeleton*, 2026.
- Delsys Incorporated, *TRIGNO® Wireless System SDK User's Guide*, MAN-025-3-5, 2021.
- Delsys Incorporated, *Trigno® Wireless Biofeedback System User's Guide*, MAN-031-1-7, 2023.
- Hermens et al., *European Recommendations for Surface ElectroMyoGraphy (SENIAM)*, 1999.
- Olah C., [*Understanding LSTM Networks*](https://colah.github.io/posts/2015-08-Understanding-LSTMs/), 2015.
- Gasq D., Cormier C., *Physiologie et évaluation de la marche*, Université de Toulouse III, 2022.
