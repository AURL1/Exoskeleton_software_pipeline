import csv
import matplotlib.pyplot as plt 
import numpy as np

#-----------CONFIG--------------------------

SENSOR_MEASUREMENTS_FILE = input('Sensor measurements file name ?')+".csv"

#-------------------------------------------

def csv_display(csv_filepath):

    time_s =[]
    roll_sensor1 =[]
    pitch_sensor1 =[]
    emg_rms_sensor1 =[]
    roll_sensor2 =[]
    pitch_sensor2 =[]
    emg_rms_sensor2 =[]
    roll_sensor3 =[]
    pitch_sensor3 =[]
    emg_rms_sensor3 =[]

    sensor1 = []
    sensor2 = []
    sensor3 = []

    sensors = [sensor1, sensor2, sensor3]

    try:

        with open(csv_filepath,  "r") as csv_file :
            csv_reader = csv.reader(csv_file, delimiter = ',')
            next(csv_reader)

            for line in csv_reader :
                time_s.append(float(line[0]))
                roll_sensor1.append(float(line[1]))
                pitch_sensor1.append(float(line[2]))
                emg_rms_sensor1.append(float(line[3]))
                roll_sensor2.append(float(line[4]))
                pitch_sensor2.append(float(line[5]))
                emg_rms_sensor2.append(float(line[6]))
                roll_sensor3.append(float(line[7]))
                pitch_sensor3.append(float(line[8]))
                emg_rms_sensor3.append(float(line[9]))


        sensor1.append(roll_sensor1)
        sensor1.append(pitch_sensor1)
        sensor1.append(emg_rms_sensor1)

        sensor2.append(roll_sensor2)
        sensor2.append(pitch_sensor2)
        sensor2.append(emg_rms_sensor2)

        sensor3.append(roll_sensor3)
        sensor3.append(pitch_sensor3)
        sensor3.append(emg_rms_sensor3)

        x_axis_EMG = np.linspace(time_s[0], time_s[-1], len(time_s))
        x_axis_IMU = np.linspace(time_s[0], time_s[-1], len(time_s))

        fig, axs = plt.subplots(3, len(sensors), figsize = (7*len(sensors),14), squeeze = False)

        for i in range (len(sensors)):
            #-----------------roll plots-----------------
            ax = axs[0, i]
            ax.plot(x_axis_IMU,  sensors[i][0], color = 'tab:blue', label = "roll")
            ax.set_xlabel("Time (s)")
            ax.set_ylabel(f"roll_sensor{i+1}")
            ax.legend(loc='upper left')

            #-----------------pitch plots-----------------
            ax = axs[1, i]
            ax.plot(x_axis_IMU, sensors[i][1], color = 'tab:red', label = "pitch")
            ax.set_xlabel("Time (s)")
            ax.set_ylabel(f"pitch_sensor{i+1}")
            ax.legend(loc='upper left')
            
            #-----------------emg plots-----------------
            ax = axs[2, i]
            ax.plot(x_axis_EMG, sensors[i][2], color = 'tab:green', label = "emg")
            ax.set_xlabel("Time (s)")
            ax.set_ylabel(f"emg_rms_sensor{i+1}")
            ax.legend(loc='upper left')
        
        plt.show()

    except FileNotFoundError:
            print(f"Error : The file '{csv_filepath}' was not found.")
            return
    
if __name__ == '__main__':
     csv_display(SENSOR_MEASUREMENTS_FILE)