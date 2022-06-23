import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as mn

from plotManager import plotManager
from utilities.utils import Utils
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

if __name__ == '__main__':
    data_path = Utils.load_config("DATASET_PATH")
    df = pd.read_csv(data_path, index_col='datetime', parse_dates=True)


    ### 1) EXPLORATORY DATA ANALYSIS ###

    plotMan = plotManager(df)

    """
    # CHECKING MISSING VALUE IN DATASET
    plotManager.check_missing_values(df)

    # DENSITY OF PROBABILITY FOR ATTRIBUTES
    plotManager.distribution_plot(df, "temp", 25, 45)
    plotManager.distribution_plot(df, "atemp", 28, 50)
    plotManager.distribution_plot(df, "humidity", 30, 110)
    plotManager.distribution_plot(df, "windspeed", 20, 60)
    plotManager.distribution_plot(df, "rentals", 20, 1010)

    # NUMBER OF RENTALS BASED ON DIFFERENT FACTORS
    plotManager.histogram_plot(df, "season")
    plotManager.histogram_plot(df, "holiday")
    plotManager.histogram_plot(df, "workingday")
    plotManager.histogram_plot(df, "weather")

    # BIVARIATE ANALYSIS FOR TARGET VARIABLE (RENTALS)
    plotManager.box_plot(df, "season")
    plotManager.box_plot(df, "hour")
    plotManager.box_plot(df, "month")
    
    # SCATTER PLOTS
    plotManager.scatter_plot(df, "temp") # Bike Rentals are observed at higher temperatures.
    plotManager.scatter_plot(df, "humidity") # Temperature being directly proportional to Humidity, Bike Rentals are making during high humidity.
    plotManager.scatter_plot(df, "windspeed") # Wind speeds increase with a greater temperature difference.Wind speed near the surface is most highly correlated with the temperature.
    plotManager.scatter_plot(df, "hour")
    plotManager.scatter_plot(df, "month")

    # ATTRIBUTE CORRELATION
    plotManager.correlation_matrix_plot(df)
    
    # It is observed that atemp and temp are highly correlated and one can be dropped to avoid multicollinearity
    df.drop('atemp',axis=1,inplace=True)
    """


    ### 2) DATA PREPROCESSING ###
