import matplotlib.pyplot as plt
import seaborn as sns
import missingno as mn


class plotManager:

    def __init__(self, df):
        self.df = df

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
            tick_lab = ['Clear', 'Mist', 'Light Snow', 'Heavy Rain']

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
            sns.boxplot(df["month"], df['rentals'])
            plt.xlabel("Month", fontsize=15)
            tick_val = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
            tick_lab = ['Jan', 'Feb', 'March', 'April', 'May', 'June', 'July', 'Aug', 'Sept', 'Oct', 'Nov', 'Dec']
            plt.xticks(tick_val, tick_lab)

        plt.ylabel("Number of Bike Rentals", fontsize=15)
        plt.show()

    def scatter_plot(df, attribute_name):
        plt.scatter(attribute_name, "rentals", data=df)
        plt.xlabel(attribute_name, fontsize=15)
        plt.ylabel("# Rentals", fontsize=15)
        plt.show()

    def check_missing_values(df):
        mn.matrix(df)
        plt.show()

    def correlation_matrix_plot(df):
        plt.figure(figsize=(15, 10))
        sns.heatmap(df.corr(), annot=True)
        plt.show()

    @staticmethod
    def forecast_plot(df):
        df_subset = df[0:7*24]

        plt.figure(figsize=(18, 6))
        plt.plot(df_subset['predicted_rentals'], label='Predicted rentals')
        plt.plot(df_subset['rentals'], label='Real rentals')
        plt.title('MLP Regressor - Forecast vs Actual Observations')
        plt.xlabel('DateTime', fontsize=15)
        plt.ylabel('Number of Bike Rentals', fontsize=15)
        plt.legend()

        labels = df_subset.index.strftime('%H:00 %a - %b %d')
        plt.xticks(df_subset.index[0::12], labels[0::12], rotation=45)
