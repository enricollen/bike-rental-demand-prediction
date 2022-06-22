import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from utilities.utils import Utils
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)


def distribution_plot(df, attribute_name, bins, max_value):
    plt.figure(figsize=(12, 7))
    ax = sns.distplot(df[attribute_name], bins=bins)
    ax.set_xlim(0, max_value)
    ax.set(xlabel='Variable ' + attribute_name)
    ax.set(ylabel='Density of Probability')
    ax.set_title('Distribution Graph', fontsize=20)
    ax.xaxis.get_label().set_fontsize(20)
    ax.yaxis.get_label().set_fontsize(20)
    plt.show()


def histogram_plot(df, attribute_name):
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

    plt.xlabel('Feature - ' + attribute_name, fontsize=15)
    plt.ylabel('Number of observations', fontsize=15)

    plt.xticks(tick_val, tick_lab)
    plt.show()


def box_plot(df, attribute_name):
    plt.figure(figsize=(8, 7))
    if attribute_name == "season":
        plt.title("Bike Rentals Based on Season Feature ", fontsize=15)
        sns.boxplot(df['season'], df['rentals'])
        plt.xlabel("Seasons", fontsize=15)
        tick_val = [0, 1, 2, 3]
        tick_lab = ['Winter', 'Spring', 'Summer', 'Fall']
        plt.xticks(tick_val, tick_lab)
    elif attribute_name == "hour":
        plt.title("Bike Rentals Based on Hour Feature", fontsize=15)
        sns.boxplot(df['hour'], df['rentals'])
        plt.xlabel("Hourly", fontsize=15)
    elif attribute_name == "month":
        plt.title("Bike Rentals on Monthly Basis", fontsize=15)
        sns.boxplot(df.index.month, df['rentals'])
        plt.xlabel("Month", fontsize=15)
        tick_val = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
        tick_lab = ['Jan', 'Feb', 'March', 'April', 'May', 'June', 'July', 'Aug', 'Sept', 'Oct', 'Nov', 'Dec']
        plt.xticks(tick_val, tick_lab)

    plt.ylabel("Number of Bike Rentals", fontsize=15)
    plt.show()


def scatter_plot(df, attribute_name):
    plt.scatter(attribute_name, "rentals", data=df)
    plt.show()


def correlation_matrix_plot(df):
    plt.figure(figsize=(15, 10))
    sns.heatmap(df.corr(), annot=True)
    plt.show()


if __name__ == '__main__':
    data_path = Utils.load_config("DATASET_PATH")
    df = pd.read_csv(data_path, index_col='datetime', parse_dates=True)

    """
    ### 1) EXPLORATORY DATA ANALYSIS ###
    
    # DENSITY OF PROBABILITY FOR ATTRIBUTES
    distribution_plot(df, "temp", 25, 45)
    distribution_plot(df, "atemp", 28, 50)
    distribution_plot(df, "humidity", 30, 110)
    distribution_plot(df, "windspeed", 20, 60)
    distribution_plot(df, "rentals", 20, 1010)
    
    # NUMBER OF RENTALS BASED ON DIFFERENT FACTORS
    histogram_plot(df, "season")
    histogram_plot(df, "holiday")
    histogram_plot(df, "workingday")
    histogram_plot(df, "weather")

    # BIVARIATE ANALYSIS FOR TARGET VARIABLE (RENTALS)
    box_plot(df, "season")
    box_plot(df, "hour")
    box_plot(df, "month")
    
    # SCATTER PLOTS
    scatter_plot(df, "temp")
    scatter_plot(df, "humidity")
    scatter_plot(df, "hour")
    scatter_plot(df, "month")

    # ATTRIBUTE CORRELATION
    correlation_matrix_plot(df)
    """

    """
    ### 2) DATA PREPROCESSING ###
    """

