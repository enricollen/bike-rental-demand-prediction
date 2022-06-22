import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from utilities.utils import Utils


def distribution_plot(df, attribute_name, bins, max_value):
    plt.figure(figsize=(12, 7))
    ax = sns.distplot(df[attribute_name], bins=bins)
    ax.set_xlim(0, max_value)
    ax.set(xlabel='Variable ' + attribute_name)
    ax.set(ylabel='Density of Probability')
    ax.set_title('Distribution Graph', fontsize=20)
    # change font size for x & y axis
    ax.xaxis.get_label().set_fontsize(20)
    ax.yaxis.get_label().set_fontsize(20)
    plt.show()


def bar_plot(df, attribute_name):
    plt.figure(figsize=[12, 6])
    if attribute_name == "season":
        plt.bar(x=df.season.value_counts().keys(), height=df.season.value_counts())
        tick_val = [1, 2, 3, 4]
        tick_lab = ['Fall', 'Spring', 'Summer', 'Winter']
    elif attribute_name == "holiday":
        plt.bar(x=df.holiday.value_counts().keys(), height=df.holiday.value_counts())
        tick_val = [0, 1]
        tick_lab = ['No', 'Yes']
    elif attribute_name == "workingday":
        plt.bar(x=df.workingday.value_counts().keys(), height=df.workingday.value_counts())
        tick_val = [0, 1]
        tick_lab = ['Holiday', 'Day is neither weekend nor holiday']
    elif attribute_name == "weather":
        plt.bar(x=df.weather.value_counts().keys(), height=df.weather.value_counts())
        tick_val = [1, 2, 3, 4]
        tick_lab = ['Condition-1', 'Condition-2', 'Condition-3', 'Condition-4']

    plt.xlabel('Feature - '+attribute_name, fontsize=15)
    plt.ylabel('Number of observations', fontsize=15)

    plt.xticks(tick_val, tick_lab)
    plt.show()


if __name__ == '__main__':
    prova = Utils.load_config("DATASET_PATH")
    df = pd.read_csv(prova, index_col='datetime', parse_dates=True)
    distribution_plot(df, "temp", 25, 45)
    distribution_plot(df, "atemp", 28, 50)
    distribution_plot(df, "humidity", 30, 110)
    distribution_plot(df, "windspeed", 20, 60)
    distribution_plot(df, "rentals", 20, 1010)
    bar_plot(df, "season")
    bar_plot(df, "holiday")
    bar_plot(df, "workingday")
    bar_plot(df, "weather")