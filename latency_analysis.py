import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np 


def features(csv_filepath):
    datas = pd.read_csv(csv_filepath)
    mask = datas["predicting_period"].notna()
    datas = datas[mask]
    median_multi_windowing_periods = datas["multi_windowing_periods"].median()
    median_guessing_periods = datas["guessing_periods"].median()
    median_predicting_periods = datas["predicting_period"].median()
    worst_case_scenario = (datas["guessing_periods"]+datas["multi_windowing_periods"]+datas["predicting_period"]).quantile(0.95)
    return(worst_case_scenario, median_multi_windowing_periods, median_guessing_periods, median_predicting_periods, datas)

def display(datas):

    fig, axes = plt.subplots(1, 3, figsize = (5 , 9), squeeze = False)

    ax = axes[0, 0]
    ax.hist(datas['multi_windowing_periods'], color = 'tab:blue', label = 'multi_windowing_periods')
    ax.set_ylabel  ("multi_windowing_periods")
    ax.legend(loc='upper left')
    ax.grid(True, alpha = 0.3)

    ax = axes[0, 1]
    ax.hist(datas['guessing_periods'], color = 'tab:red', label = 'guessing_periods')
    ax.set_ylabel  ("guessing_periods")
    ax.legend(loc='upper left')
    ax.grid(True, alpha = 0.3)

    ax = axes[0, 2]
    ax.hist(datas['predicting_period'], color = 'tab:green', label = 'predicting_period')
    ax.set_ylabel  ("predicting_period")
    ax.legend(loc='upper left')
    ax.grid(True, alpha = 0.3)

    plt.show()

if __name__ == "__main__":
    csv_file = input("csv filepath ?")
    (worst_case_scenario, median_multi_windowing_periods, median_guessing_periods, median_predicting_periods, datas) = features(csv_file)
    print(f" worst_case_scenario : {worst_case_scenario} "
          f"\n median_multi_windowing_periods :{median_multi_windowing_periods}"
          f"\n median_guessing_periods : {median_guessing_periods}"
          f"\n median_predicting_periods : {median_predicting_periods}" 
          f"\n datas : {datas}")
    display(datas)