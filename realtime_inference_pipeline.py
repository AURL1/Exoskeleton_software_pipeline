"""
realtime_inference_pipeline.py

Goal: the ONLINE loop. While the exosuit is being worn, this script:
    1. Continuously acquires and filters IMU + EMG signals in two
       background threads (imu_acquisition_loop, emg_acquisition_loop).
    2. Every ~14 ms, extracts the same windowed features used during
       training and asks the three trained SVMs for their (averaged)
       class probabilities (multi_windowing + predict_gait_phase).
    3. Feeds a rolling sequence of those probabilities into the trained
       LSTM, which predicts the gait phase expected at t + delta.
    4. Converts that predicted phase into a finite-state-machine (FSM)
       state, and sends the matching torque command to the CubeMars
       motor over CAN bus (motor_command), with a watchdog thread that
       falls back to PASSIVE if commands stop arriving in time.

This mirrors the OFFLINE pipeline (angles_and_EMG_measurements.py ->
datas_preparation_SVM.py -> SVM.py -> datas_preparation_RNN.py ->
RNN.py) but runs everything live, sample by sample, instead of reading
from a CSV.
"""

from angles_and_EMG_measurements import build_sensors, design_filters_imu, design_filters_emg, design_antialias_filter
from scipy.signal import sosfilt, lfilter
import numpy as np
from collections import deque
import threading
import pandas as pd
import time
import joblib
from RNN import LSTM
import torch
import glob
from enum import Enum
import can

# -----------CONFIG--------------------------
NUM_SENSORS = 3  # change this number for the number of sensors used

FIRST_SENSOR_SLOT = 1
SENSOR_MODE = 39

CHANNEL_RANGE_EMG = (0, NUM_SENSORS - 1)

FS_IMU = 74                     # IMU sampling frequency, in Hz
FS_EMG = 2000                   # EMG sampling frequency, in Hz
SAMPLES_PER_READ_IMU = 2        # how many IMU samples come in per read() call
SAMPLES_PER_READ_EMG = 200      # how many EMG samples come in per read() call
DECIMATION_FACTOR = int(round(FS_EMG / FS_IMU))  # how many EMG samples correspond to one IMU sample

WINDOW_SEC = 5                  # length (in seconds) of the rolling buffers kept in memory
RENDER_FPS = 30
RENDER_INTERVAL_MS = int(1000 / RENDER_FPS)

BUFFER_LEN_IMU = int(WINDOW_SEC * FS_IMU)
BUFFER_LEN_EMG = FS_EMG * WINDOW_SEC
BUFFER_LEN_EMG_DS = BUFFER_LEN_IMU   # "DS" = downsampled EMG (RMS, brought down to the IMU's rate)

SAVE_INTERVAL_SEC = 0.5

LOWPASS_CUTOFF_IMU_HZ = 4.0

RMS_WINDOW_MS = 250
RMS_WINDOW_SAMPLES = int(RMS_WINDOW_MS * 1e-3 * FS_EMG)

# Same three window sizes used to train the three SVMs offline - must
# stay in sync with datas_preparation_SVM.py.
FIRST_WINDOW_SIZE = 13
SECOND_WINDOW_SIZE = 26
THIRD_WINDOW_SIZE = 37

FS_IMU = 74
DELAY_PROGRAMS_IN_SECONDS = 0.058       # measured reaction delay of the full pipeline + motor
DELTA_SAMPLES = round(FS_IMU * DELAY_PROGRAMS_IN_SECONDS)  # that delay, in samples
SEQUENCE_LENGTH = 2 * DELTA_SAMPLES     # length of the sequence fed to the LSTM (note: 2x delta here, vs 1x in datas_preparation_RNN.py)

# The 6 possible states of the assist finite-state machine.
states = Enum('states', [('INIT'), ('PASSIVE'), ('STANDING_STILL'), ('DUAL_THRUST_PHASE'), ('SWING_PHASE'), ('DOUBLE_SUPPORT_LANDING_PHASE'), ('SINGLE_LEG_SUPPORT_PHASE')])
CONFIDENCE_THRESHOLD = 0.85     # minimum LSTM confidence required to trust a phase prediction
MAX_UNCERTAIN = 37             # after this many consecutive low-confidence predictions, fall back to PASSIVE

MOTOR_CAN_ID = 1
MIT_MODE_ID = 8            # CubeMars MIT (force control) mode
MOTOR_DISABLE_MODE_ID = 15
SET_ORIGIN_MODE = 5

# CubeMars AK60-6 V3.0 parameter ranges, used to convert real physical
# values into the fixed-point integers the motor's CAN protocol expects.
P_MIN = -12.56
P_MAX = 12.56
V_MIN = -60.0
V_MAX = 60.0
T_MIN = -12.0
T_MAX = 12.0
KP_MIN = 0
KP_MAX = 500
KD_MIN = 0
KD_MAX = 5.0

# Bit width used to encode each parameter in the CAN message.
P_DES_BITS = 16
V_DES_BITS = 12
T_FF_BITS = 12
KP_BITS = 12
KD_BITS = 12

# For each FSM state: the (position, velocity, Kp, Kd, feed-forward
# torque) command sent to the motor. Only DUAL_THRUST_PHASE and
# SWING_PHASE actually assist (t_ff = 3.6 Nm, Kp = Kd = 0 -> pure
# feed-forward torque, no position/velocity control mixed in). Every
# other state sends a "do nothing" command.
state_configuration = {states.INIT: (0, 0, 0, 0, 0),
                        states.PASSIVE: (0, 0, 0, 0, 0),
                        states.STANDING_STILL: (0, 0, 0, 0, 0),
                        states.DUAL_THRUST_PHASE: (0, 0, 0, 0, 3.6),
                        states.SWING_PHASE: (0, 0, 0, 0, 3.6),
                        states.DOUBLE_SUPPORT_LANDING_PHASE: (0, 0, 0, 0, 0),
                        states.SINGLE_LEG_SUPPORT_PHASE: (0, 0, 0, 0, 0)}
# Maps the LSTM's predicted label strings (as they were labelled in
# Label Studio) back to the matching FSM state.
label_to_state_correspondence = {"Swing phase": states.SWING_PHASE,
                                  "double-support landing phase": states.DOUBLE_SUPPORT_LANDING_PHASE,
                                  "Single-leg support phase": states.SINGLE_LEG_SUPPORT_PHASE,
                                  "Dual-thrust phase": states.DUAL_THRUST_PHASE}

STANDING_WINDOW = 37
EMG_STANDING_THRESHOLD = 0.09
ANGULAR_VELOCITY_STANDING_THRESHOLD = 3
# -------------------------------------------


def imu_acquisition_loop(imus, buf_roll, buf_pitch, stop_event, buf_emg, buf_emg_ds_rms, sos_lp_imu, zi_roll, zi_pitch, sos_aa, zi_aa_rms):
    """
    Background thread: continuously reads IMU angles, low-pass filters
    them, and also computes a downsampled EMG RMS envelope (from the raw
    EMG samples the OTHER thread is filling into buf_emg) so both
    signal types end up available at the IMU's sampling rate.

    All the zi_* arguments are filter "state" arrays (see
    angles_and_EMG_measurements.py / the scipy sosfilt docs): they let
    each filter pick up exactly where it left off between chunks,
    instead of resetting (and producing a glitch) every loop iteration.
    """
    imus[0].start()
    imus[0].prepare_buffer()

    decim_counter = [0] * NUM_SENSORS

    print(f"[IMU] Streaming IMU started - mode {SENSOR_MODE}, slots {FIRST_SENSOR_SLOT} to {FIRST_SENSOR_SLOT + NUM_SENSORS - 1}")

    sample_counter = 0

    while not stop_event.is_set():
        try:
            roll_chunks = []
            pitch_chunks = []
            for i, imu in enumerate(imus):
                # -------reading IMU of one sensor-------------
                roll_chunk_raw, pitch_chunk_raw = imu.read_euler()
                roll_chunk_smooth, zi_roll[i] = sosfilt(sos_lp_imu, roll_chunk_raw, zi=zi_roll[i])
                pitch_chunk_smooth, zi_pitch[i] = sosfilt(sos_lp_imu, pitch_chunk_raw, zi=zi_pitch[i])
                buf_roll[i].extend(roll_chunk_smooth)
                buf_pitch[i].extend(pitch_chunk_smooth)
                roll_chunks.append(roll_chunk_smooth)
                pitch_chunks.append(pitch_chunk_smooth)

            emg_ds_latest_rms = []

            for ch in range(NUM_SENSORS):
                emg_array = np.array(buf_emg[ch])
                n_emg_per_cycle = SAMPLES_PER_READ_IMU * DECIMATION_FACTOR

                # We need a bit of EMG history BEFORE the newest samples
                # too, so the moving-RMS window below has enough context
                # to compute a value for every new sample (not just the
                # very last one). n_context is exactly how many EMG
                # samples that requires.
                n_context = n_emg_per_cycle + RMS_WINDOW_SAMPLES - 1
                if len(emg_array) >= n_context:
                    segment_ctx = emg_array[-n_context:]
                else:
                    # Not enough EMG history yet (e.g. right at startup)
                    # -> pad the missing part with zeros instead of
                    # crashing or waiting.
                    pad = np.zeros(n_context - len(emg_array))
                    segment_ctx = np.concatenate([pad, emg_array])

                # Moving RMS : mean of the squared values on the window, then square rooted
                sq = segment_ctx ** 2
                kernel = np.ones(RMS_WINDOW_SAMPLES) / RMS_WINDOW_SAMPLES
                rms_segment = np.sqrt(np.convolve(sq, kernel, mode='valid'))

                # Anti-aliasing and decimation on the RMS signal
                rms_aa, zi_aa_rms[ch] = sosfilt(sos_aa, rms_segment, zi=zi_aa_rms[ch])
                decimated_rms = rms_aa[::DECIMATION_FACTOR]
                buf_emg_ds_rms[ch].extend(decimated_rms)
                emg_ds_latest_rms.append(float(decimated_rms[-1]) if len(decimated_rms) > 0 else 0.0)

        except IOError:
            print("[IMU] Disconnection detected, stop.")
            stop_event.set()
            break

    imus[0].stop()
    print("[IMU] Threading stopped")


def emg_acquisition_loop(emg, buf_emg, stop_event):
    """
    Background thread: continuously reads raw EMG samples, applies the
    bandpass + notch filters (same design as in
    angles_and_EMG_measurements.py) and rectifies them (abs value),
    filling buf_emg - which imu_acquisition_loop then reads from to
    compute the RMS envelope above.
    """
    print(f"[EMG] Streaming EMG started ")

    # ---Construction of the filters for the EMG datas------
    sos_bp, (b_notch, a_notch) = design_filters_emg(FS_EMG)
    n_sections_bp = sos_bp.shape[0]
    zi_bp = np.zeros((NUM_SENSORS, n_sections_bp, 2))
    zi_notch = np.zeros((NUM_SENSORS, max(len(b_notch), len(a_notch)) - 1))

    while not stop_event.is_set():
        try:
            # ------reading EMG of one sensor--------------
            emg_chunk = emg.read()
            for ch in range(NUM_SENSORS):
                x_raw = emg_chunk[ch]
                y_bp, zi_bp[ch] = sosfilt(sos_bp, x_raw, zi=zi_bp[ch])
                y_bp, zi_notch[ch] = lfilter(b_notch, a_notch, y_bp, zi=zi_notch[ch])
                y_rect = np.abs(y_bp)
                buf_emg[ch].extend(y_rect)

        except IOError:
            print("[EMG] Disconnection detected, stop.")
            stop_event.set()
            break
    print("[EMG] Threading stopped.")


def window_layout(window_size, buff_roll, buff_pitch, buff_emg_ds_rms):
    """
    Real-time equivalent of window_layout_SVM / window_layout_RNN: take
    the LAST `window_size` samples currently sitting in the rolling
    buffers, and compute the same 5 features per sensor (mean, std, min,
    max, growth rate) used to train the SVMs.

    Returns None if there isn't yet enough data in the buffers to fill a
    full window (e.g. right after startup) - the caller uses this to
    skip that cycle instead of crashing.
    """

    all_features = []
    features_sensors = {}
    roll_array = [np.array(buff_roll[ch]) for ch in range(NUM_SENSORS)]
    pitch_array = [np.array(buff_pitch[ch]) for ch in range(NUM_SENSORS)]
    emg_array = [np.array(buff_emg_ds_rms[ch]) for ch in range(NUM_SENSORS)]

    for ch in range(NUM_SENSORS):

        if len(roll_array[ch]) < window_size:
            return (None)
        else:

            roll_mean = float(roll_array[ch][-window_size:].mean())
            roll_std = float(roll_array[ch][-window_size:].std())
            roll_min = float(min(roll_array[ch][-window_size:]))
            roll_max = float(max(roll_array[ch][-window_size:]))
            roll_growth_rate = float((roll_array[ch][-window_size:][-1] - roll_array[ch][-window_size:][0]) / window_size)

            pitch_mean = float(pitch_array[ch][-window_size:].mean())
            pitch_std = float(pitch_array[ch][-window_size:].std())
            pitch_min = float(min(pitch_array[ch][-window_size:]))
            pitch_max = float(max(pitch_array[ch][-window_size:]))
            pitch_growth_rate = float((pitch_array[ch][-window_size:][-1] - pitch_array[ch][-window_size:][0]) / window_size)

            emg_mean = float(emg_array[ch][-window_size:].mean())
            emg_std = float(emg_array[ch][-window_size:].std())
            emg_max = float(max(emg_array[ch][-window_size:]))
            # emg_squared_sum = float(sum(np.square(csv_datas[f"emg_rms_sensor{i}"][index:index+window_size]))/window_size)
            emg_growth_rate = float((emg_array[ch][-window_size:][-1] - emg_array[ch][-window_size:][0]) / window_size)

            features_sensors.update({f"roll_mean{ch+1}": roll_mean,
                                      f"roll_std{ch+1}": roll_std,
                                      f"roll_min{ch+1}": roll_min,
                                      f"roll_max{ch+1}": roll_max,
                                      f"roll_growth_rate{ch+1}": roll_growth_rate,

                                      f"pitch_mean{ch+1}": pitch_mean,
                                      f"pitch_std{ch+1}": pitch_std,
                                      f"pitch_min{ch+1}": pitch_min,
                                      f"pitch_max{ch+1}": pitch_max,
                                      f"pitch_growth_rate{ch+1}": pitch_growth_rate,

                                      f"emg_mean{ch+1}": emg_mean,
                                      f"emg_std{ch+1}": emg_std,
                                      f"emg_max{ch+1}": emg_max,
                                      #  f"emg_squared_sum{ch}":emg_squared_sum,
                                      f"emg_growth_rate{ch+1}": emg_growth_rate})

    all_features.append(features_sensors)

    all_features = pd.DataFrame(all_features)
    print(f"all features (DataFrame):\n{all_features}")

    return (all_features)


def multi_windowing(window_size1, window_size2, window_size3, buff_roll, buff_pitch, buff_emg_ds_rms):
    """
    Run window_layout() once per window size (13/26/37). If any one of
    them isn't ready yet (buffers too short), the whole call returns
    None so the caller waits for the next cycle instead of using
    mismatched/partial data.
    """
    all_features_window_size1 = window_layout(window_size1, buff_roll, buff_pitch, buff_emg_ds_rms)
    all_features_window_size2 = window_layout(window_size2, buff_roll, buff_pitch, buff_emg_ds_rms)
    all_features_window_size3 = window_layout(window_size3, buff_roll, buff_pitch, buff_emg_ds_rms)
    if ((all_features_window_size1 is None) or (all_features_window_size2 is None) or (all_features_window_size3 is None)):
        return (None)
    else:
        return (all_features_window_size1, all_features_window_size2, all_features_window_size3)


def predict_gait_phase(first_features_file, second_features_file, third_features_file, trained_pipeline1, trained_pipeline2, trained_pipeline3):
    """
    Real-time equivalent of crossing_prediction() in
    datas_preparation_RNN.py: ask each of the three trained SVMs for its
    class probabilities on its matching window size, then average the
    three predictions together into one probability vector.
    """
    first_prediction = trained_pipeline1.predict_proba(first_features_file)
    second_prediction = trained_pipeline2.predict_proba(second_features_file)
    third_prediction = trained_pipeline3.predict_proba(third_features_file)
    predictions = np.vstack((first_prediction, second_prediction, third_prediction))
    crossed_predictions = np.mean(predictions, axis=0)
    # recognised_label_index = np.argmax(crossed_predictions)
    # recognised_label = trained_pipeline1.classes_[recognised_label_index]
    return (crossed_predictions)


def design_phase_prediction_lstm():
    """
    Reload the LSTM architecture (from RNN.py) and load the most
    recently saved trained weights (phase_prediction_lstm*.pt) into it,
    ready for inference (.eval() disables dropout).
    """
    lstm = LSTM()
    best_lstm = torch.load(glob.glob(f"phase_prediction_lstm*.pt")[0])
    lstm.load_state_dict(best_lstm)
    lstm.eval()
    return (lstm)


def standing_still_detection(buff_pitch, buff_emg_ds_rms):
    """
    Heuristic (not ML-based) check for whether the wearer is currently
    standing still: true only if BOTH the maximum pitch angular
    velocity AND the maximum EMG activity, across all sensors, stay
    under their respective thresholds over the last STANDING_WINDOW
    samples. Returns False if there isn't enough buffered data yet to
    check.
    """
    emg_mean_sensors = []
    max_pitch_velocity_sensors = []
    for ch in range(NUM_SENSORS):
        if len(buff_pitch[ch]) < STANDING_WINDOW or len(buff_emg_ds_rms[ch]) < STANDING_WINDOW:
            return (False)
        else:
            pitch_array = np.array(buff_pitch[ch])[-STANDING_WINDOW:]
            emg_array = np.array(buff_emg_ds_rms[ch])[-STANDING_WINDOW:]
            emg_mean_sensors.append(np.mean(emg_array))
            # np.diff gives the change between consecutive samples;
            # multiplying by FS_IMU converts "change per sample" into
            # "change per second", i.e. an angular velocity.
            pitch_velocity_sensors = np.abs(np.diff(pitch_array) * FS_IMU)
            max_pitch_velocity_sensors.append(max(pitch_velocity_sensors))
    emg_mean_max = max(emg_mean_sensors)
    pitch_velocity = max(max_pitch_velocity_sensors)
    return (pitch_velocity < ANGULAR_VELOCITY_STANDING_THRESHOLD and emg_mean_max < EMG_STANDING_THRESHOLD)


def next_state(current_state, predicted_label, confidence, uncertain_count, standing_still):
    """
    The finite-state-machine transition function.

    Priority order:
      1. If standing_still_detection() says the wearer is standing
         still, force STANDING_STILL regardless of what the LSTM says.
      2. Otherwise, if the LSTM's confidence is high enough, trust its
         prediction and switch to the matching state; reset the
         uncertain-prediction counter.
      3. Otherwise (low confidence), stay in the current state and bump
         the uncertain-prediction counter - unless that counter has hit
         MAX_UNCERTAIN, in which case fall back to PASSIVE for safety.
    """
    if standing_still:
        return (states.STANDING_STILL, 0)
    if confidence >= CONFIDENCE_THRESHOLD:
        new_count = 0
        new_state = label_to_state_correspondence[predicted_label]
        return (new_state, new_count)
    else:
        if uncertain_count >= MAX_UNCERTAIN:
            new_count = 0
            new_state = states.PASSIVE
            return (new_state, new_count)
        else:
            new_count = uncertain_count + 1
            new_state = current_state
            return (new_state, new_count)


def float_to_uint(value, x_min, x_max, bits):
    """
    Quantize a physical float value (e.g. a torque in Nm) into an
    unsigned integer that fits in `bits` bits, for the CubeMars CAN
    protocol. The value is first clipped to [x_min, x_max] so an
    out-of-range value can't silently wrap around to a wrong command.
    """
    value_clipped = np.clip(value, x_min, x_max)
    uint = int((value_clipped - x_min) / (x_max - x_min) * ((1 << bits) - 1))
    return (uint)


def pack_mit_command(p_des, v_des, Kp, Kd, t_ff):
    """
    Pack the 5 MIT-mode command values (position, velocity, Kp, Kd,
    feed-forward torque) into the 8-byte CAN payload expected by the
    CubeMars AK60-6 V3.0 in MIT mode. Byte layout: Kp and Kd first, then
    position/velocity/torque - this is the V3.0-specific ordering (see
    project notes: different from the older Ben Katz / Mini-Cheetah
    ordering).
    """

    Kp_int = float_to_uint(Kp, KP_MIN, KP_MAX, KP_BITS)
    byte_0 = Kp_int >> 4
    byte_1_top_part = (Kp_int & 0xF) << 4

    Kd_int = float_to_uint(Kd, KD_MIN, KD_MAX, KD_BITS)
    byte_1 = byte_1_top_part | (Kd_int >> 8)
    byte_2 = Kd_int & 0xFF

    p_int = float_to_uint(p_des, P_MIN, P_MAX, P_DES_BITS)
    byte_3 = p_int >> 8
    byte_4 = p_int & 0xFF

    v_int = float_to_uint(v_des, V_MIN, V_MAX, V_DES_BITS)
    byte_5 = v_int >> 4
    byte_6_top_part = (v_int & 0xF) << 4

    t_ff_int = float_to_uint(t_ff, T_MIN, T_MAX, T_FF_BITS)
    byte_6 = byte_6_top_part | (t_ff_int >> 8)
    byte_7 = t_ff_int & 0xFF

    eight_bytes_message = bytearray([byte_0, byte_1, byte_2, byte_3, byte_4, byte_5, byte_6, byte_7])
    return (eight_bytes_message)


def motor_command(motor_timestamp, state, bus, lock):
    """
    Look up the (p_des, v_des, Kp, Kd, t_ff) tuple for `state`, pack it
    into a CAN message, and send it to the motor. `lock` protects the
    CAN bus from being written to by two threads at once (this function
    is called both from recording_loop and from motor_watchdog).
    motor_timestamp[0] is updated on every successful send, which is
    what the watchdog thread checks to detect a stalled main loop.
    """
    (p_des, v_des, Kp, Kd, t_ff) = state_configuration[state]
    command = pack_mit_command(p_des, v_des, Kp, Kd, t_ff)
    msg = can.Message(arbitration_id=(MIT_MODE_ID << 8) | MOTOR_CAN_ID, data=command, is_extended_id=True)
    try:
        with lock:
            bus.send(msg)
            motor_timestamp[0] = time.perf_counter()
            print(f"Message sent on {bus.channel_info}")
    except can.CanError:
        print("Message NOT sent")


def deactivate_motor(bus, lock):
    """
    Send the CubeMars "disable motor" mode frame (empty payload), used
    during shutdown to make sure the motor stops applying any torque.
    """
    try:
        deactivation_msg = can.Message(arbitration_id=(MOTOR_DISABLE_MODE_ID << 8) | MOTOR_CAN_ID, data=bytearray(0), is_extended_id=True)
        with lock:
            bus.send(deactivation_msg)
    except can.CanError:
        print("Deactivation message NOT sent")


def motor_watchdog(last_timestamp, bus, lock, stop_event):
    """
    Background safety thread: every 20 ms, check how long it's been
    since the last successful motor_command(). If more than 0.1s has
    passed (meaning the main recording_loop has stalled, crashed, or is
    running too slowly), force the motor into PASSIVE rather than
    letting it keep applying a stale command.
    """
    while not stop_event.is_set():
        time.sleep(0.02)
        current_time = time.perf_counter()
        if current_time - last_timestamp[0] > 0.1:
            motor_command(last_timestamp, states.PASSIVE, bus, lock)


def start_threads(buff_roll, buff_pitch, buff_emg, buff_emg_ds_rms, stop_event, motor_timestamp, bus, lock):
    """
    Build the sensor objects and filters, then start the three
    background threads (IMU acquisition, EMG acquisition, motor
    watchdog) as daemons and return their thread handles so
    recording_loop can join() them on shutdown.
    """

    imus, emg = build_sensors()

    sos_lp_imu = design_filters_imu(FS_IMU)
    n_sections_lp = sos_lp_imu.shape[0]
    zi_roll = np.zeros((NUM_SENSORS, n_sections_lp, 2))
    zi_pitch = np.zeros((NUM_SENSORS, n_sections_lp, 2))

    sos_aa = design_antialias_filter(fs_high=FS_EMG, decimation_factor=DECIMATION_FACTOR)
    n_sections_aa = sos_aa.shape[0]
    zi_aa_rms = np.zeros((NUM_SENSORS, n_sections_aa, 2))

    imu_thread = threading.Thread(
        target=imu_acquisition_loop,
        args=(imus, buff_roll, buff_pitch, stop_event, buff_emg, buff_emg_ds_rms, sos_lp_imu, zi_roll, zi_pitch, sos_aa, zi_aa_rms),
        daemon=True  # the thread stops automatically if the main program crashes
    )

    emg_thread = threading.Thread(
        target=emg_acquisition_loop,
        args=(emg, buff_emg, stop_event),
        daemon=True  # the thread stops automatically if the main program crashes
    )

    motor_thread = threading.Thread(
        target=motor_watchdog,
        args=(motor_timestamp, bus, lock, stop_event),
        daemon=True  # the thread stops automatically if the main program crashes
    )

    imu_thread.start()
    emg_thread.start()
    motor_thread.start()

    return (imu_thread, emg_thread, motor_thread)


def recording_loop(buff_roll, buff_pitch, buff_emg_ds_rms, buff_recognised_walking_phases,
                    stop_event, imu_thread, emg_thread, motor_thread,
                    first_trained_pipeline, second_trained_pipeline, third_trained_pipeline,
                    lstm, label_encoder, bus, motor_timestamp, lock):
    """
    The main real-time loop. On every ~14 ms cycle:
      1. Extract SVM features for the 3 window sizes and get the
         averaged SVM probability vector (multi_windowing + predict_gait_phase).
      2. Push that probability vector into buff_recognised_walking_phases
         (a fixed-length rolling buffer of length SEQUENCE_LENGTH).
      3. Once that buffer is full, feed the whole sequence to the LSTM to
         get a predicted future phase + confidence, run the standing-still
         heuristic, and compute the next FSM state (next_state).
      4. Send the corresponding motor command for the current FSM state.

    Timing information for every cycle (windowing time, SVM time, LSTM
    time) is recorded and saved to timestamps.csv at the end, which is
    how DELAY_PROGRAMS_IN_SECONDS was originally measured.

    On Ctrl+C or any unhandled situation, the `finally` block makes sure
    the motor is set back to PASSIVE, then fully deactivated, and that
    all background threads are properly joined before exiting.
    """
    try:
        timestamps = []
        current_state = states.INIT
        uncertain_count = 0

        motor_command(motor_timestamp, states.PASSIVE, bus, lock)

        while not stop_event.is_set():
            time.sleep(0.014)
            start_multi_windowing = time.perf_counter()
            results = multi_windowing(FIRST_WINDOW_SIZE, SECOND_WINDOW_SIZE, THIRD_WINDOW_SIZE, buff_roll, buff_pitch, buff_emg_ds_rms)
            end_multi_windowing = time.perf_counter()
            multi_windowing_period = (end_multi_windowing - start_multi_windowing)
            if results is None:
                # Buffers not full enough yet this cycle - log timing and
                # move on to the next cycle without a prediction.
                timestamps.append({"multi_windowing_periods": multi_windowing_period,
                                    "guessing_periods": None, "predicting_period": None})
                continue
            else:
                (features_windowsize13, features_windowsize26, features_windowsize37) = results
                start_categorising = time.perf_counter()
                recognised_labels = predict_gait_phase(features_windowsize13, features_windowsize26, features_windowsize37,
                                                        first_trained_pipeline, second_trained_pipeline, third_trained_pipeline)
                end_categorising = time.perf_counter()
                categorising_period = (end_categorising - start_categorising)

                buff_recognised_walking_phases.append(recognised_labels)
                print(recognised_labels)
                if len(buff_recognised_walking_phases) == SEQUENCE_LENGTH:
                    # The rolling sequence buffer is full: run the LSTM.
                    start_predicting = time.perf_counter()
                    array_recognised_walking_phases = np.array(list(buff_recognised_walking_phases))
                    tensor_recognised_walking_phases = torch.from_numpy(np.float32(array_recognised_walking_phases))
                    # unsqueeze(0) adds a batch dimension of size 1, since
                    # the LSTM expects a batch of sequences, not a single
                    # bare sequence.
                    tensor_recognised_walking_phases = tensor_recognised_walking_phases.unsqueeze(0)
                    with torch.no_grad():
                        logits = lstm(tensor_recognised_walking_phases)
                        tensor_proba_logits = torch.softmax(logits, dim=1)
                        confidence = (tensor_proba_logits).max().item()
                        prediction = torch.argmax(logits, dim=1)
                        prediction = prediction.numpy()
                        predicted_string_label = label_encoder.inverse_transform(prediction)[0]
                        print(f"predicted_string_label : {predicted_string_label}",
                              f"proba logits : {tensor_proba_logits}")
                        bool_standing_still = standing_still_detection(buff_pitch, buff_emg_ds_rms)
                        (current_state, uncertain_count) = next_state(current_state, predicted_string_label, confidence, uncertain_count, bool_standing_still)
                    end_predicting = time.perf_counter()
                    predicting_period = (end_predicting - start_predicting)
                    timestamps.append({"multi_windowing_periods": multi_windowing_period,
                                        "guessing_periods": categorising_period,
                                        "predicting_period": predicting_period})
                else:
                    # Sequence buffer still filling up - no LSTM
                    # prediction yet this cycle, state stays unchanged.
                    timestamps.append({"multi_windowing_periods": multi_windowing_period,
                                        "guessing_periods": categorising_period,
                                        "predicting_period": None})
                motor_command(motor_timestamp, current_state, bus, lock)

    except KeyboardInterrupt:
        print("Keyboard interrupt.")

    finally:
        # Shutdown sequence: stop all background threads, bring the
        # motor safely back down to PASSIVE, wait one cycle, then fully
        # disable it before exiting, and save the timing log.
        stop_event.set()
        imu_thread.join(timeout=3)
        emg_thread.join(timeout=3)
        motor_thread.join(timeout=3)
        motor_command(motor_timestamp, states.PASSIVE, bus, lock)
        time.sleep(0.014)
        deactivate_motor(bus, lock)
        pd.DataFrame(timestamps).to_csv("timestamps.csv")
        print("Program finalised.")


if __name__ == "__main__":
    # Rolling buffers shared between the acquisition threads and the
    # main recording_loop.
    buff_roll = [deque(maxlen=BUFFER_LEN_IMU) for _ in range(NUM_SENSORS)]
    buff_pitch = [deque(maxlen=BUFFER_LEN_IMU) for _ in range(NUM_SENSORS)]
    buff_emg = [deque(maxlen=BUFFER_LEN_EMG) for _ in range(NUM_SENSORS)]
    buff_emg_ds_rms = [deque(maxlen=BUFFER_LEN_EMG_DS) for _ in range(NUM_SENSORS)]
    buff_recognised_walking_phases = deque(maxlen=SEQUENCE_LENGTH)

    # Reload the three trained SVM pipelines (one per window size), same
    # filenames as in datas_preparation_RNN.py.
    SVM_results = {"trained_pipeline_window13": joblib.load(f"pipe_features_windowsize_of_13_session20260612_143920 - Copie.pkl"),
                   "trained_pipeline_window26": joblib.load(f"pipe_features_windowsize_of_26_session20260612_143920 - Copie.pkl"),
                   "trained_pipeline_window37": joblib.load(f"pipe_features_windowsize_of_37_session20260612_143920 - Copie.pkl")}

    stop_event = threading.Event()
    label_encoder = joblib.load(glob.glob("label_encoder*.pkl")[0])
    my_phase_prediction_lstm = design_phase_prediction_lstm()
    motor_threading_lock = threading.Lock()

    # `with can.Bus(...)` ensures the CAN bus is properly closed even if
    # an exception occurs inside the block.
    with can.Bus(interface='slcan', channel='COM3', bitrate=1000000) as bus:
        motor_timestamp = [time.perf_counter()]
        (imu_thread, emg_thread, motor_thread) = start_threads(buff_roll, buff_pitch, buff_emg, buff_emg_ds_rms, stop_event, motor_timestamp, bus, motor_threading_lock)
        recording_loop(buff_roll, buff_pitch, buff_emg_ds_rms, buff_recognised_walking_phases,
                       stop_event, imu_thread, emg_thread, motor_thread,
                       SVM_results["trained_pipeline_window13"], SVM_results["trained_pipeline_window26"], SVM_results["trained_pipeline_window37"],
                       my_phase_prediction_lstm, label_encoder, bus, motor_timestamp, motor_threading_lock)