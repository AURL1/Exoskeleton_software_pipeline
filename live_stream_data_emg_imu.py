#This code has been built over Nicklas code 

from pytrigno import TrignoEMG, TrignoAccel
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import sosfilt, butter, iirnotch, lfilter 
from collections import deque
from math import ceil
import threading
import matplotlib.animation as animation

#-------CONFIG------------------------------
CHANNEL_RANGE_EMG=(0,1)             #since each sensors only has 1 EMG channel, this is also the number of sensors
CHANNEL_RANGE_IMU = (0,17)          #number of IMU channel we use, knowing that we use both the accelerometer and the gyrometer
FS_EMG=2000
FS_IMU=148
WINDOW_SEC = 1
SAMPLES_PER_READ_EMG=200
SAMPLES_PER_READ_IMU=15

RENDER_FPS = 30                   #enable to refresh the plots every 30 ms, indepedently of the data sampling rate
RENDER_INTERVAL_MS = int(1000/RENDER_FPS)

BUFFER_LEN_EMG = FS_EMG * WINDOW_SEC
BUFFER_LEN_IMU = FS_IMU * WINDOW_SEC
#-------------------------------------------

def design_filters_emg(fs):

    #Band pass 20-450 Hz (4th order)
    sos_bp = butter(4, [20,450], btype='bandpass', fs=fs, output='sos')

    #Notch 50 Hz (Q=30)
    b_notch, a_notch = iirnotch(50, 30, fs=fs)

    return sos_bp, (b_notch, a_notch)


def design_filters_imu(fs):
     
     #Enveloppe low-pass 10 Hz (2nd order)
    sos_env = butter(2, 10, btype='lowpass', fs=fs, output='sos')

    return sos_env
     

def stream_data_filt():
    emg = TrignoEMG(
        channel_range=CHANNEL_RANGE_EMG,
        samples_per_read=SAMPLES_PER_READ_EMG,
        units='mV',
        host='localhost'
    )

    imu = TrignoAccel(
        channel_range=CHANNEL_RANGE_IMU,
        samples_per_read=SAMPLES_PER_READ_IMU,
        host='localhost'
    )

    num_sensors = CHANNEL_RANGE_EMG[1] - CHANNEL_RANGE_EMG[0] + 1

    print((f"Streaming {num_sensors} Sensors"))

    #Filters 
    sos_bp, (b_notch, a_notch) = design_filters_emg(FS_EMG)
    sos_env = design_filters_imu(FS_IMU)

    #States for IIR filters, per channel
    # sosfilt needed shape (n_section, 2) for each channel
    n_sections_bp = sos_bp.shape[0]
    n_sections_env = sos_env.shape[0]

    zi_bp = np.zeros((num_sensors, n_sections_bp, 2))

    zi_env_acc_X = np.zeros((num_sensors, n_sections_env,2))
    zi_env_acc_Y = np.zeros((num_sensors, n_sections_env,2))
    zi_env_acc_Z = np.zeros((num_sensors, n_sections_env,2))

    zi_env_gyr_X = np.zeros((num_sensors, n_sections_env,2))
    zi_env_gyr_Y = np.zeros((num_sensors, n_sections_env,2))
    zi_env_gyr_Z = np.zeros((num_sensors, n_sections_env,2))

    zi_notch = np.zeros((num_sensors, max(len(b_notch), len(a_notch)) - 1))


    #Ring buffers (enveloppe for plotting)
    buffers_emg = [deque(np.zeros(BUFFER_LEN_EMG), maxlen=BUFFER_LEN_EMG) for _ in range(num_sensors)]

    buffers_imu_acc_X = [deque(np.zeros(BUFFER_LEN_IMU), maxlen=BUFFER_LEN_IMU) for _ in range(num_sensors)]
    buffers_imu_acc_Y = [deque(np.zeros(BUFFER_LEN_IMU), maxlen=BUFFER_LEN_IMU) for _ in range(num_sensors)]
    buffers_imu_acc_Z = [deque(np.zeros(BUFFER_LEN_IMU), maxlen=BUFFER_LEN_IMU) for _ in range(num_sensors)]

    buffers_imu_gyr_X = [deque(np.zeros(BUFFER_LEN_IMU), maxlen=BUFFER_LEN_IMU) for _ in range(num_sensors)]
    buffers_imu_gyr_Y = [deque(np.zeros(BUFFER_LEN_IMU), maxlen=BUFFER_LEN_IMU) for _ in range(num_sensors)]
    buffers_imu_gyr_Z = [deque(np.zeros(BUFFER_LEN_IMU), maxlen=BUFFER_LEN_IMU) for _ in range(num_sensors)]

    #------------PLOT SETUP-----------------------------------

    plt.ion()

    #-----------Plots EMG----------

    n_cols_emg = 2
    n_rows_emg = ceil( num_sensors / n_cols_emg )

    fig_emg, axes_emg = plt.subplots(n_rows_emg, n_cols_emg, figsize = (12,6), sharex=True, squeeze = False)
    axes_emg = axes_emg.flatten()

    x_axis_emg = np.arange(BUFFER_LEN_EMG) / FS_EMG

    lines_emg = []

    for ch in range(num_sensors):
        ax = axes_emg[ch]
        line_emg, = ax.plot(x_axis_emg, np.zeros(BUFFER_LEN_EMG), lw=1)
        ax.set_title(f"Sensor {ch}")
        ax.set_ylim(-0.5,0.5)
        ax.set_xlim(0,WINDOW_SEC)
        lines_emg.append(line_emg)

    #---Plots accelerometer IMU---

    n_cols_imu_acc = 3
    n_imu_acc_plots = max(num_sensors * n_cols_imu_acc, n_cols_imu_acc)
    n_rows_imu_acc = ceil(n_imu_acc_plots / n_cols_imu_acc)

    fig_imu_acc, axes_imu_acc = plt.subplots(n_rows_imu_acc, n_cols_imu_acc, figsize = (12,6), sharex=True, squeeze = False)
    axes_imu_acc = axes_imu_acc.flatten()

    x_axis_imu_acc = np.arange(BUFFER_LEN_IMU) / FS_IMU
    lines_imu_acc_X, lines_imu_acc_Y, lines_imu_acc_Z = [], [], []
    coordinates = ['X', 'Y', 'Z']
    color = ['tab:blue','tab:orange','tab:green']

    for ch in range(num_sensors):
        for i,coord in enumerate(coordinates):
            ax_idx_acc = ch * n_cols_imu_acc + i
            ax = axes_imu_acc[ax_idx_acc]   
            line_acc, = ax.plot(x_axis_imu_acc, np.zeros(BUFFER_LEN_IMU), lw=1, color = color[i])
            ax.set_title(f"Sensor {ch} - ACC {coord}")
            ax.set_ylim(-0.5,0.5)   
            ax.set_xlim(0,WINDOW_SEC)
            if i == 0:
                lines_imu_acc_X.append(line_acc)
            if i == 1:
                lines_imu_acc_Y.append(line_acc)
            if i == 2:
                lines_imu_acc_Z.append(line_acc)

    #---Plots gryrometer IMU-----

    n_cols_imu_gyr = 3
    n_imu_gyr_plots = max(num_sensors * n_cols_imu_gyr, n_cols_imu_gyr)
    n_rows_imu_gyr = ceil(n_imu_gyr_plots / n_cols_imu_gyr)

    fig_imu_gyr, axes_imu_gyr = plt.subplots(n_rows_imu_gyr, n_cols_imu_gyr, figsize = (12,6), sharex=True, squeeze = False)
    axes_imu_gyr = axes_imu_gyr.flatten()

    x_axis_imu_gyr = np.arange(BUFFER_LEN_IMU) / FS_IMU
    lines_imu_gyr_X, lines_imu_gyr_Y, lines_imu_gyr_Z = [], [], []

    for ch in range(num_sensors):
        for i,coord in enumerate(coordinates):
            ax_idx_gyr = ch * n_cols_imu_gyr + i
            ax = axes_imu_gyr[ax_idx_gyr]   
            line_gyr, = ax.plot(x_axis_imu_gyr, np.zeros(BUFFER_LEN_IMU), lw=1, color = color[i])
            ax.set_title(f"Sensor {ch} - GYR {coord}") 
            ax.set_ylim(-0.5,0.5)   
            ax.set_xlim(0,WINDOW_SEC)
            if i == 0:
                lines_imu_gyr_X.append(line_gyr)
            if i == 1:
                lines_imu_gyr_Y.append(line_gyr)
            if i == 2:
                lines_imu_gyr_Z.append(line_gyr)

    #Hide unused axes 
    for i in range (num_sensors, len(axes_emg)):
        axes_emg[i].axis("off")

    for i in range (num_sensors * n_cols_imu_acc, len(axes_imu_acc)):
         axes_imu_acc[i].axis("off")

    for i in range (num_sensors * n_cols_imu_gyr, len(axes_imu_gyr)):
         axes_imu_gyr[i].axis("off")

    plt.tight_layout()
    plt.show()

    #------------------------------------------------------------------------

    fig_emg.canvas.draw()
    fig_imu_acc.canvas.draw()
    fig_imu_gyr.canvas.draw()

    stop_event = threading.Event()

    def acquisition_loop():

        #request the master connexion to the server (to use bot the IMU and the EMG), 
        #because you only need to call for one or the other to get the master connexion to both
        emg.start()
        print("Streaming EMG and IMU started.. CTRL+C to stop.")

        _sosfilt   = sosfilt
        _lfilter   = lfilter
        _np_abs    = np.abs

        while not stop_event.is_set():
            #get the EMG data 
            new_chunk_emg = emg.read()
            #get the IMU data 
            new_chunk_imu = imu.read()

            #Process each channel independently 
            for ch in range(num_sensors):

                #--------------------EMG-----------------------------

                #new EMG acquisitions
                x_raw_emg = new_chunk_emg[ch]
                
                # ---- 1)Band pass 20-450 Hz (IIR) ----
                y_bp_emg, zi_bp[ch] = sosfilt(sos_bp, x_raw_emg, zi=zi_bp[ch])

                # ---- 2) Optional : notch 50 Hz ---
                #If line noise is bad, uncomment it:
                y_bp_emg, zi_notch[ch] = lfilter(b_notch, a_notch, y_bp_emg, zi = zi_notch[ch])

                # --- Rectify ---
                y_rect_emg = np.abs(y_bp_emg)

                #Append to ring buffer (use enveloppe for plotting)
                buffers_emg[ch].extend(y_rect_emg)
                
                #----------------------------------------------------

                #--------------------IMU Accelerometer-----------------------------

                #new accelerometer acquisitions (x, y , z coordinates)
                x_raw_imu_acc = new_chunk_imu[ch*3]
                y_raw_imu_acc = new_chunk_imu[ch*3 + 1]
                z_raw_imu_acc = new_chunk_imu[ch*3 + 2]

                #----Low pass filter (Enveloppe 10 Hz)----
                y_env_imu_acc_X, zi_env_acc_X[ch] = sosfilt(sos_env, x_raw_imu_acc, zi=zi_env_acc_X[ch])
                y_env_imu_acc_Y, zi_env_acc_Y[ch] = sosfilt(sos_env, y_raw_imu_acc, zi=zi_env_acc_Y[ch])
                y_env_imu_acc_Z, zi_env_acc_Z[ch] = sosfilt(sos_env, z_raw_imu_acc, zi=zi_env_acc_Z[ch])

                #Append to ring buffer (use enveloppe for plotting)
                buffers_imu_acc_X[ch].extend(y_env_imu_acc_X)
                buffers_imu_acc_Y[ch].extend(y_env_imu_acc_Y)
                buffers_imu_acc_Z[ch].extend(y_env_imu_acc_Z)

                #------------------------------------------------------------------

                #--------------------IMU Gyrometer---------------------------------

                #new gyrometer acquisitions (x, y , z coordinates)
                x_raw_imu_gyr = new_chunk_imu[ch*3 + 3]
                y_raw_imu_gyr = new_chunk_imu[ch*3 + 4]
                z_raw_imu_gyr = new_chunk_imu[ch*3 + 5]

                #----Low pass filter (Enveloppe 10 Hz)----
                y_env_imu_gyr_X, zi_env_gyr_X[ch] = sosfilt(sos_env, x_raw_imu_gyr, zi=zi_env_gyr_X[ch])
                y_env_imu_gyr_Y, zi_env_gyr_Y[ch] = sosfilt(sos_env, y_raw_imu_gyr, zi=zi_env_gyr_Y[ch])
                y_env_imu_gyr_Z, zi_env_gyr_Z[ch] = sosfilt(sos_env, z_raw_imu_gyr, zi=zi_env_gyr_Z[ch])

                #Append to ring buffer (use enveloppe for plotting)
                buffers_imu_gyr_X[ch].extend(y_env_imu_gyr_X)
                buffers_imu_gyr_Y[ch].extend(y_env_imu_gyr_Y)
                buffers_imu_gyr_Z[ch].extend(y_env_imu_gyr_Z)

            #------------------------------------------------------------------

        emg.stop()
        
    acq_thread = threading.Thread(target=acquisition_loop, daemon=True)
    acq_thread.start()

    #Update plots 
    def update_emg(frame):
        for ch in range(num_sensors):
            lines_emg[ch].set_ydata(buffers_emg[ch])
        return lines_emg
    
    def update_imu_acc(frame):
        for ch in range(num_sensors):
            lines_imu_acc_X[ch].set_ydata(buffers_imu_acc_X[ch])
            lines_imu_acc_Y[ch].set_ydata(buffers_imu_acc_Y[ch])
            lines_imu_acc_Z[ch].set_ydata(buffers_imu_acc_Z[ch])
        return lines_imu_acc_X + lines_imu_acc_Y + lines_imu_acc_Z
    
    def update_imu_gyr(frame):
        for ch in range(num_sensors):
            lines_imu_gyr_X[ch].set_ydata(buffers_imu_gyr_X[ch])
            lines_imu_gyr_Y[ch].set_ydata(buffers_imu_gyr_Y[ch])
            lines_imu_gyr_Z[ch].set_ydata(buffers_imu_gyr_Z[ch])
        return lines_imu_gyr_X + lines_imu_gyr_Y + lines_imu_gyr_Z
        
    ani_emg = animation.FuncAnimation(
        fig_emg, update_emg,
        interval = RENDER_INTERVAL_MS,
        blit = True,
        cache_frame_data = False
    )

    ani_imu_acc = animation.FuncAnimation(
        fig_imu_acc, update_imu_acc,
        interval = RENDER_INTERVAL_MS,
        blit = True,
        cache_frame_data = False
    )

    ani_imu_gyr = animation.FuncAnimation(
        fig_imu_gyr, update_imu_gyr,
        interval = RENDER_INTERVAL_MS,
        blit = True,
        cache_frame_data = False
    )

    try: 
        plt.show(block=True)

    except KeyboardInterrupt:
        print("Stopped by user.")

    finally:
        stop_event.set()
        acq_thread.join(timeout=2)
        plt.ioff()
        plt.show()

if __name__ == "__main__":
     stream_data_filt()