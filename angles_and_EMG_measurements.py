"""
This code is designed to record EMG and IMU data from Delsys Trigno Avanti sensors and display the data in real time, and save it to a CSV file.

Previous script: none (first script to be run)
Next script: resampling.py

For this script to work, you must launch the Trigno control unit via localhost first, then you must attach the sensors to the patient, and finally run the code. 
The number of sensors is configurable via NUM_SENSORS (currently set to 3).

Architecture : 
Three threads run in parallel: one for the IMU, one for the EMG, and one for periodic backups. 
They share buffers protected by a lock, and all shut down cleanly via a common stop signal.
A low-pass filter is applied to the IMU signals. A band-pass filter, a notch filter (50 Hz, to filter out mains noise) and an anti-aliasing filter are applied to the EMG signal.

This script generates a CSV file saved under the name 'session' followed by the date and time of the recording.
The CSV file contains one time column and 3 columns per sensor : two for the IMU (roll and pitch angles) and one for the EMG.
"""

from pytrigno import IM, TrignoEMG
import numpy as np
import matplotlib.pyplot as plt 
import matplotlib.animation as animation 
from collections import deque
import threading 
from scipy.signal import sosfilt, butter, iirnotch, lfilter, sosfilt_zi
import pandas as pd
from datetime import datetime
import logging 

#-----------CONFIG--------------------------
NUM_SENSORS = 3 #change this number for the number of sensors used

TRIGNO_TIMEOUT_S = 10
ANTIALIAS_MARGIN = 0.9
ANTIALIAS_ORDER=8
EMG_BANDPASS_ORDER=4
EMG_BANDPASS_HZ=[20,450]
NOTCH_FREQ_HZ=50
NOTCH_Q=30
IMU_LOWPASS_ORDER=2
STATES_PER_SOS_SECTION=2

ROLL_YLIM=180
PITCH_YLIM=90
EMG_YLIM=0.5

FIRST_SENSOR_SLOT = 1
SENSOR_MODE = 39

CHANNEL_RANGE_EMG=(0,NUM_SENSORS - 1)

FS_IMU = 74
FS_EMG=2000
SAMPLES_PER_READ_IMU = 2
SAMPLES_PER_READ_EMG=200
DECIMATION_FACTOR = int(round(FS_EMG/FS_IMU))

WINDOW_SEC = 5
RENDER_FPS = 30
RENDER_INTERVAL_MS = int(1000/RENDER_FPS)

BUFFER_LEN_IMU = int(WINDOW_SEC*FS_IMU)
BUFFER_LEN_EMG = FS_EMG * WINDOW_SEC
BUFFER_LEN_EMG_DS = BUFFER_LEN_IMU

SAVE_INTERVAL_SEC = 0.5

LOWPASS_CUTOFF_IMU_HZ = 4.0

RMS_WINDOW_MS = 250
RMS_WINDOW_SAMPLES = int(RMS_WINDOW_MS * 1e-3 * FS_EMG) 
#-------------------------------------------

def build_sensors():
    """Create and configure the IMU and EMG sensor objects (one IM per slot, one shared TrignoEMG).

    Returns:
        imus: list of IM objects, one per sensor slot.
        emg: a single TrignoEMG object reading all EMG channels at once.
    """
    imus = []
    for i in range (NUM_SENSORS):
        slot = FIRST_SENSOR_SLOT + i
        imu = IM ( 
            sensor_slot = slot,
            samples_per_read = SAMPLES_PER_READ_IMU,
            host = 'localhost',
            timeout = TRIGNO_TIMEOUT_S
        )
        imu.set_mode(SENSOR_MODE)
        imus.append(imu)

    emg = TrignoEMG(
        channel_range = CHANNEL_RANGE_EMG,
        samples_per_read = SAMPLES_PER_READ_EMG,
        units = 'mV',
        host = 'localhost'
    )

    return imus, emg

def design_antialias_filter(fs_high, decimation_factor):
    """Build the low-pass filter applied before decimating the EMG RMS signal, to avoid aliasing."""
    fs_low = fs_high / decimation_factor
    f_cutoff = (fs_low / 2) * ANTIALIAS_MARGIN
    sos_aa = butter(ANTIALIAS_ORDER, f_cutoff, btype = 'lowpass', fs=fs_high, output='sos')

    return sos_aa


def design_filters_emg(fs):
    """Build the two filters applied to raw EMG: a band-pass (keeps the muscle-activity frequency band)
    and a notch (removes 50 Hz mains/electrical noise)."""
    #Band pass 20-450 Hz (4th order)
    sos_bp = butter(EMG_BANDPASS_ORDER, EMG_BANDPASS_HZ, btype='bandpass', fs=fs, output='sos')
    #Notch 50 Hz (Q=30)
    b_notch, a_notch = iirnotch(NOTCH_FREQ_HZ, NOTCH_Q, fs=fs)

    return sos_bp, (b_notch, a_notch)

def design_filters_imu(fs, cutoff = LOWPASS_CUTOFF_IMU_HZ):
    """Build the low-pass filter used to smooth the raw roll/pitch angles from the IMU."""
    #Butterworth 2nd order low pass for Euler angles smoothing
    sos_lp = butter(IMU_LOWPASS_ORDER, cutoff, btype='lowpass', fs=fs, output='sos')

    return sos_lp

def saving_loop(record_buffer, record_lock, stop_event, filepath):
    """Thread target: every SAVE_INTERVAL_SEC, move whatever rows the IMU thread has
    accumulated in record_buffer into the CSV file at `filepath`, then keep going
    until stop_event is set, at which point it does one last flush before exiting."""
    first_write = True

    #Periodically empty record_buffer to store it into a csv file
    while not stop_event.is_set():
        stop_event.wait(timeout=SAVE_INTERVAL_SEC)

        #we take the lock the less time possible
        #we steal the lines accumulated and we free them immediately after
        with record_lock:
            if not record_buffer:
                continue
            rows_to_save = record_buffer.copy()
            record_buffer.clear()

        df_chunk = pd.DataFrame(rows_to_save)

        if first_write :
            df_chunk.to_csv(filepath, index = False, mode = 'w')
            first_write = False
        else :
            df_chunk.to_csv(filepath, index = False, mode ='a', header = False)
        
        print(f"[SAVE] {len(rows_to_save)} lines saved in {filepath}")

    #Final saving of the remaining files after the program stops
    with record_lock:
        if record_buffer:
            df_chunk = pd.DataFrame(record_buffer)
            df_chunk.to_csv(filepath, index = False, mode = 'a' if not first_write else 'w', header = first_write)
            record_buffer.clear()
            print(f"[SAVE] Final flush : {len(df_chunk)} lines saved")

    print("[SAVE] Saving thread stopped.")

def imu_acquisition_loop(imus, buf_roll,buf_pitch, stop_event, buf_emg, buf_emg_ds_rms, record_buffer, record_lock, 
                         t0, sos_lp_imu, zi_roll, zi_pitch, sos_aa, zi_aa_rms):
    """Thread target: the main acquisition loop, running at FS_IMU (~74 Hz).

    On every iteration it:
      1. reads and low-pass-filters the raw roll/pitch angles for each IMU,
      2. turns the raw EMG buffer into a decimated RMS signal (see the RMS/
         anti-aliasing block below) so it can be plotted and saved alongside
         the IMU data at the same rate,
      3. appends one row per IMU sample to record_buffer (protected by
         record_lock), which saving_loop() later writes to the CSV.

    Stops on the first IOError, or when stop_event is set by another thread.
    """
    imus[0].start()
    imus[0].prepare_buffer()
    imus[1].prepare_buffer()
    imus[2].prepare_buffer()
    
    print(f"[IMU] Streaming IMU started - mode {SENSOR_MODE}, slots {FIRST_SENSOR_SLOT} to {FIRST_SENSOR_SLOT + NUM_SENSORS - 1}")

    roll0 = []
    pitch0 = []

    for imu in imus:
        roll_val, pitch_val = imu.read_euler()
        roll0.append(roll_val[0])
        pitch0.append(pitch_val[0])

    for i in range(NUM_SENSORS):
        zi_roll[i] = sosfilt_zi(sos_lp_imu)*roll0[i]
        zi_pitch[i] = sosfilt_zi(sos_lp_imu)*pitch0[i]

    while not stop_event.is_set():
        try:
            roll_chunks = []
            pitch_chunks = []
            for i, imu in enumerate(imus):
                #-------reading IMU of one sensor-------------
                roll_chunk_raw, pitch_chunk_raw = imu.read_euler()
                roll_chunk_smooth, zi_roll[i] = sosfilt(sos_lp_imu, roll_chunk_raw, zi=zi_roll[i])
                pitch_chunk_smooth, zi_pitch[i] = sosfilt(sos_lp_imu, pitch_chunk_raw, zi=zi_pitch[i])
                buf_roll[i].extend(roll_chunk_smooth)
                buf_pitch[i].extend(pitch_chunk_smooth)
                roll_chunks.append(roll_chunk_smooth)
                pitch_chunks.append(pitch_chunk_smooth)
            
            t_now = datetime.now().timestamp() - t0

            emg_ds_latest_rms = []

            for ch in range(NUM_SENSORS):
                emg_array = np.array(buf_emg[ch])
                n_emg_per_cycle = SAMPLES_PER_READ_IMU * DECIMATION_FACTOR

                n_context = n_emg_per_cycle + RMS_WINDOW_SAMPLES - 1
                if len(emg_array) >= n_context:
                    segment_ctx = emg_array[-n_context:]
                else:
                    pad = np.zeros(n_context - len(emg_array))
                    segment_ctx = np.concatenate([pad, emg_array])

                # Moving RMS : mean of the squared values on the window, then square rooted
                sq = segment_ctx ** 2
                kernel = np.ones(RMS_WINDOW_SAMPLES) / RMS_WINDOW_SAMPLES
                rms_segment = np.sqrt(np.convolve(sq, kernel, mode = 'valid'))

                # Anti-aliasing and decimation on the RMS signal
                rms_aa, zi_aa_rms[ch] = sosfilt(sos_aa, rms_segment, zi = zi_aa_rms[ch])
                decimated_rms = rms_aa[::DECIMATION_FACTOR]
                buf_emg_ds_rms[ch].extend(decimated_rms)
                emg_ds_latest_rms.append(float(decimated_rms[-1]) if len(decimated_rms)>0 else 0.0)

            for s in range(SAMPLES_PER_READ_IMU):
                
                #we take the values from the sensors at a frequency of 74 Hz
                row = {"time_s": t_now - ((SAMPLES_PER_READ_IMU-1-s)/FS_IMU)}

                for ch in range(NUM_SENSORS):
                    slot = FIRST_SENSOR_SLOT + ch
                    row[f"roll_sensor{slot}"] = float(roll_chunks[ch][s])
                    row[f"pitch_sensor{slot}"] = float(pitch_chunks[ch][s])
                    row[f"emg_rms_sensor{slot}"] = emg_ds_latest_rms[ch]

                with record_lock:
                    record_buffer.append(row)
        except IOError:
            logging.error("[IMU] Error detected, stop.")
            stop_event.set()
            break

    imus[0].stop()
    print("[IMU] Threading stopped")

def emg_acquisition_loop(emg, buf_emg, stop_event):
    """Thread target: reads raw EMG at FS_EMG (2000 Hz), applies the band-pass + notch
    filters (see design_filters_emg), rectifies the signal (absolute value), and
    stores the result in buf_emg. imu_acquisition_loop later reads from buf_emg
    to compute the decimated RMS used for plotting/saving.

    Stops on the first IOError, or when stop_event is set by another thread.
    """
    print(f"[EMG] Streaming EMG started ")

    #---Construction of the filters for the EMG datas------
    sos_bp, (b_notch, a_notch) = design_filters_emg(FS_EMG)
    n_sections_bp = sos_bp.shape[0]
    zi_bp = np.zeros((NUM_SENSORS, n_sections_bp, STATES_PER_SOS_SECTION))
    zi_notch = np.zeros((NUM_SENSORS, max(len(b_notch), len(a_notch)) - 1))

    while not stop_event.is_set():
        try:
            #------reading EMG of one sensor--------------
            emg_chunk = emg.read()
            for ch in range(NUM_SENSORS):
                x_raw = emg_chunk[ch]
                y_bp, zi_bp[ch] = sosfilt(sos_bp, x_raw, zi=zi_bp[ch])
                y_bp, zi_notch[ch] = lfilter(b_notch, a_notch, y_bp, zi = zi_notch[ch])
                y_rect = np.abs(y_bp)
                buf_emg[ch].extend(y_rect)

        except IOError:
            logging.error("[EMG] Error detected, stop.")
            stop_event.set()
            break
    print("[EMG] Threading stopped.")

def stream_angles_and_EMG():
    """Entry point. Sets up the shared buffers/filters, connects to the sensors,
    builds the live matplotlib figure, starts the three worker threads
    (IMU acquisition, EMG acquisition, CSV saving), and runs the plot's
    animation loop until the window is closed or the user interrupts (Ctrl+C),
    at which point all threads are asked to stop and joined before exiting.
    """

    #-----------ring buffers initialized to zero--------------------------
    buf_roll = [deque(np.zeros(BUFFER_LEN_IMU), maxlen = BUFFER_LEN_IMU) for _ in range(NUM_SENSORS)]
    buf_pitch = [deque(np.zeros(BUFFER_LEN_IMU), maxlen = BUFFER_LEN_IMU) for _ in range(NUM_SENSORS)]
    buf_emg = [deque(np.zeros(BUFFER_LEN_EMG), maxlen=BUFFER_LEN_EMG) for _ in range(NUM_SENSORS)]
    buf_emg_ds_rms = [deque(np.zeros(BUFFER_LEN_EMG_DS), maxlen=BUFFER_LEN_EMG_DS) for _ in range(NUM_SENSORS)]

    print("[INIT] Connexion to the TCU and sensors configuration...")
    imus, emg = build_sensors()

    sos_lp_imu = design_filters_imu(FS_IMU)
    n_sections_lp = sos_lp_imu.shape[0]
    zi_roll = np.zeros((NUM_SENSORS,n_sections_lp, STATES_PER_SOS_SECTION))
    zi_pitch = np.zeros((NUM_SENSORS,n_sections_lp, STATES_PER_SOS_SECTION)) 

    sos_aa = design_antialias_filter(fs_high = FS_EMG, decimation_factor = DECIMATION_FACTOR)
    n_sections_aa = sos_aa.shape[0]
    zi_aa_rms = np.zeros((NUM_SENSORS, n_sections_aa, STATES_PER_SOS_SECTION))

    record_buffer = []
    record_lock = threading.Lock()
    t0 = datetime.now().timestamp()
    session_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_filepath = f"session{session_str}.csv"
    print(f"[SAVE] The datas will be saved in {csv_filepath}")

    #-----------temporal axis-----------
    x_axis_imu = np.linspace(-WINDOW_SEC, 0, BUFFER_LEN_IMU)
    x_axis_emg_ds = np.linspace(-WINDOW_SEC, 0, BUFFER_LEN_EMG_DS)

    fig, axes = plt.subplots(3, NUM_SENSORS, figsize = (5 * NUM_SENSORS, 9), squeeze = False)

    fig.suptitle(
        f"Roll / Pitch / EMG - {NUM_SENSORS} sensor(s)"
        f"slots {FIRST_SENSOR_SLOT} - {FIRST_SENSOR_SLOT + NUM_SENSORS - 1}",
        fontsize = 13
    )

    lines_roll = []
    lines_pitch = []
    lines_emg_rms = []

    for ch in range (NUM_SENSORS):
        #---------------Roll (line 0)---------------
        slot = FIRST_SENSOR_SLOT + ch
        ax = axes[0, ch]
        line_roll, = ax.plot(x_axis_imu, np.zeros(BUFFER_LEN_IMU), color = 'tab:blue', lw = 1, label = 'Roll (smoothed)')
        ax.set_title(f"Sensor {slot}")
        ax.set_ylabel  ("Roll(°)")
        ax.set_xlabel("Time (s)")
        ax.set_ylim(-ROLL_YLIM,ROLL_YLIM)
        ax.axhline(0,color='k', lw= 0.5, ls='--')
        ax.legend(loc='upper left')
        ax.grid(True, alpha = 0.3)
        lines_roll.append(line_roll)
        
        #---------------Pitch (line 1)---------------
        ax = axes[1, ch]
        line_pitch, = ax.plot(x_axis_imu, np.zeros(BUFFER_LEN_IMU), color = 'tab:orange', lw = 1, label = 'Pitch (smoothed)')
        ax.set_ylabel  ("Pitch(°)")
        ax.set_xlabel("Time (s)")
        ax.set_ylim(-PITCH_YLIM,PITCH_YLIM)
        ax.axhline(0,color='k', lw= 0.5, ls='--')
        ax.legend(loc='upper left')
        ax.grid(True, alpha = 0.3)
        lines_pitch.append(line_pitch)

        #---------------EMG (line 2)---------------
        ax = axes[2, ch]
        line_emg_rms, = ax.plot(x_axis_emg_ds, np.zeros(BUFFER_LEN_EMG_DS), lw=1, color='tab:red', label = f"EMG RMS {RMS_WINDOW_MS} ms (~{FS_IMU} Hz)")
        ax.set_title(f"EMG - Sensor {slot}")
        ax.set_ylabel(f"EMG (mV)")
        ax.set_xlabel("Time (s)")
        ax.set_ylim(0,EMG_YLIM)
        ax.set_xlim(-WINDOW_SEC, 0)
        ax.legend(loc='upper left', fontsize = 8)
        ax.grid(True, alpha = 0.3)
        lines_emg_rms.append(line_emg_rms)

    plt.tight_layout()

    stop_event  = threading.Event()

    #------Acquisition IMU thread--------
    imu_thread = threading.Thread(
        target= imu_acquisition_loop,
        args = (
            imus, buf_roll, buf_pitch, stop_event,
            buf_emg, buf_emg_ds_rms,
            record_buffer, record_lock,
            t0,
            sos_lp_imu, zi_roll, zi_pitch,
            sos_aa, zi_aa_rms,
        ),
        daemon = True # the thread stops automatically if the main program crashes 
    )

    #------Acquisition EMG thread--------
    emg_thread = threading.Thread(
        target= emg_acquisition_loop,
        args = (emg, buf_emg, stop_event),
        daemon = True # the thread stops automatically if the main program crashes 
    )

    #------Saving CSV thread--------
    save_thread = threading.Thread(
        target = saving_loop,
        args = (record_buffer, record_lock, stop_event, csv_filepath),
        daemon = True
    )

    imu_thread.start()
    emg_thread.start()
    save_thread.start()

    def update(frame):
        for ch in range(NUM_SENSORS):
            lines_roll[ch].set_ydata(np.array(buf_roll[ch]))
            lines_pitch[ch].set_ydata(np.array(buf_pitch[ch]))
            lines_emg_rms[ch].set_ydata(np.array(buf_emg_ds_rms[ch]))

        return lines_roll + lines_pitch + lines_emg_rms
    
    ani = animation.FuncAnimation(
        fig,
        update,
        interval = RENDER_INTERVAL_MS,
        blit = True,
        cache_frame_data = False # no caching : the data changes with every frame 
    )

    try:
        plt.show(block = True)
    except KeyboardInterrupt:
        print("[MAIN] User interrupt detected.")
    finally:
        stop_event.set()
        imu_thread.join(timeout = 3)
        emg_thread.join(timeout = 3)
        save_thread.join(timeout = 10)
        plt.ioff()
        plt.close('all')
        print("[MAIN] Program finalised.")

if __name__ == "__main__":
    stream_angles_and_EMG()