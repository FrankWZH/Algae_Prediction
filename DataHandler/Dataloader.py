import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from pandas import DataFrame
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline


class DataReader:
    def __init__(self, file_path: str, column_names):
        self.data = pd.read_csv(file_path)  # loading energy data
        self.col_names = column_names
        self.data.drop(self.col_names, axis=1, inplace=True)
        # Columns to be removed due to all 0 or Nan values
        # self.check_nans_dups(self.data)

    def interpolate_nan(self):
        self.data.interpolate(method='linear', limit_direction='forward', inplace=True)

    @staticmethod
    def check_nans_dups(df_input):
        """
        Checking Nans and duplicates in each column
        :param df_input:
        :return:
        """
        print('Number of Nans in each column :')
        print(df_input.isnull().sum())
        print()
        print(f'Number of duplicates in the dataframe : {df_input.duplicated().sum()}')

    @staticmethod
    def set_time_to_index(data: DataFrame, from_index: str, to_index: str):
        """
        Change the indexed column to datetime type and set it as index
        :param data: Dataframe
        :param from_index: The index of the dataframe
        :param to_index: The index of the dataframe
        :return: Changed dataframe
        """
        data[to_index] = pd.to_datetime(data[from_index], utc=True)
        data = data.set_index(to_index)
        return data

    @staticmethod
    def feat_corr(input_df, title_name):
        # Choose data column
        numeric_df = input_df.select_dtypes(include=[np.number])
        corr = numeric_df.corr()
        plt.figure(figsize=(15, 12))
        # Draw the thermal graph
        g = sns.heatmap(corr, annot=True, cmap="RdYlGn", vmin=-1, vmax=1)
        plt.title(title_name)
        return plt.show()

    def drop_column(self, column, axis=1, inplace=True):
        self.data.drop(column, axis=axis, inplace=inplace)

    @staticmethod
    def reshaping(x):
        reshaped_x = x.reshape(-1, x.shape[1] * x.shape[2])
        return reshaped_x

    @staticmethod
    def windowing(x_input, y_input, history_size):
        data = []
        labels = []
        for i in range(history_size, len(y_input)):
            data.append(x_input[i - history_size: i, :])
            labels.append(y_input[i])

        return np.array(data), np.array(labels).reshape(-1, 1)

    @staticmethod
    def apply_pca(x_input, cum_variance, if_apply):
        if if_apply:

            pca = PCA(n_components=cum_variance)
            # make pipeline to first standardize then apply PCA on data
            scaler_pca = make_pipeline(MinMaxScaler(), pca)
            x_pca = scaler_pca.fit(x_input).transform(x_input)

            return x_pca

        else:

            return np.array(x_input)

# col_names = ['generation fossil coal-derived gas', 'generation fossil oil shale', 'generation fossil peat',
#              'generation geothermal', 'generation hydro pumped storage aggregated', 'generation marine',
#              'generation wind offshore', 'forecast wind offshore eday ahead', 'forecast solar day ahead',
#              'forecast wind onshore day ahead']
#
# Energy_Data = DataReader(r'../energy_dataset.csv', col_names)
#
# Energy_Data.data = Energy_Data.set_time_to_index(Energy_Data.data, "time", "time")
#
# Energy_Data.data.interpolate(method='linear', limit_direction='forward', inplace=True)
#
# Energy_Data.data.isnull().sum()
#
# # Energy_Data.feat_corr(Energy_Data.data, 'Energy Data Feature Correlation')
#
#
# Energy_Data.data["generation fossil total"] = (Energy_Data.data['generation fossil hard coal'] +
#                                                Energy_Data.data['generation fossil brown coal/lignite'])
# Energy_Data.data.drop(['generation fossil hard coal', 'generation fossil brown coal/lignite'], axis=1, inplace=True)
#
# # Energy_Data.feat_corr(Energy_Data.data, 'Processed Energy Data Feature Correlation')
#
#
# col_drop_name = ['weather_id', 'weather_main', 'weather_description', 'weather_icon', 'temp_min', 'temp_max']
# # Loading weather data
#
# Weather_Data = DataReader(r'../weather_features.csv', col_drop_name)
#
# # Weather_Data.feat_corr(Weather_Data.data, 'Feature Correlation')
#
# Weather_Data.data.reset_index().drop_duplicates()
# Weather_Data.set_time_to_index(Weather_Data.data, "dt_iso", "time")
# Weather_Data.drop_column("dt_iso")
#
# Weather_Data.data.describe().round(2)
#
# Weather_Data.data.loc[Weather_Data.data['pressure'] > 1080,  'pressure'] = np.nan
# Weather_Data.data.loc[Weather_Data.data['pressure'] < 870,  'pressure'] = np.nan
# Weather_Data.data.loc[Weather_Data.data['wind_speed'] > 113, 'wind_speed'] = np.nan
#
# Weather_Data.data.interpolate(method='linear', limit_direction='forward', inplace=True)
#
# # be sure to drop rain_3h
# Weather_Data.data.drop(['rain_3h'], axis=1, inplace=True)
#
#
# # So number of samples in each group of city is not consistent! probably it has duplicates!
# # print(f'Number of samples in df_energy is {Weather_Data.data.shape[0]}')
#
# # city_list = Weather_Data.data['city_name'].unique()
# # grouped_weather = Weather_Data.data.groupby('city_name')
# #
# # for city in city_list:
# #     print(f'Number of samples in df_weather in {city} is {grouped_weather.get_group(city).shape[0]}')
#
# df_weather_cleaned = (Weather_Data.data.reset_index().drop_duplicates(subset=['time', 'city_name'], keep='first').
#                       set_index('time'))
#
#
# # Now the number of samples in each group is the same! So we can concat weather dataframe with energy dataframe.
# print(f'Number of samples in df_energy is {Energy_Data.data.shape[0]}')
#
# city_list = Weather_Data.data['city_name'].unique()
# grouped_weather = df_weather_cleaned.groupby('city_name')
#
# for city in city_list:
#     print(f'Number of samples in df_weather in {city} is {grouped_weather.get_group(city).shape[0]}')
#
# df_weather_all_cities = [grouped_weather.get_group(x) for x in grouped_weather.groups]
#
# df_weather_energy = Energy_Data.data
#
# for df_city in df_weather_all_cities:
#     city_name = df_city.iloc[0]['city_name'].replace(' ', '')
#     df_temp_city = df_city.add_suffix(f'_{city_name}')
#     df_weather_energy = pd.concat([df_weather_energy, df_temp_city], axis=1)
#     df_weather_energy = df_weather_energy.drop(f'city_name_{city_name}', axis=1)
#
#
# # Add these attributes to the data as new columns
# df_weather_energy['hour'] = df_weather_energy.index.map(lambda x: x.hour)
# df_weather_energy['weekday'] = df_weather_energy.index.map(lambda x: x.weekday())
# df_weather_energy['month'] = df_weather_energy.index.map(lambda x: x.month)
# df_weather_energy['year'] = df_weather_energy.index.map(lambda x: x.year)

# plt.figure(figsize=(12, 6))
# # plot total load actual for two weeks duration
# plt.plot(Energy_Data.data['total load actual'][: 24 * 7 * 2])
# plt.xlabel('Time')
# plt.ylabel('total load actual')
#
# fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 8))
#
# # select the columns to plot
# columns_to_plot = ['pressure', 'wind_speed', 'rain_1h', 'rain_3h']
#
# # loop through the subplots and plot each column
# for i, ax in enumerate(axes.flat):
#     if i < len(columns_to_plot):
#         ax.plot(Weather_Data.data.index, Weather_Data.data[columns_to_plot[i]])
#         ax.set_title(columns_to_plot[i])
#     else:
#         ax.set_visible(False)
#
# plt.tight_layout()  # adjust the spacing between subplots
# plt.show()  # display the plot
#
# fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 8))
#
# # select the columns to plot
# columns_to_plot = ['pressure', 'wind_speed', 'rain_1h', 'rain_3h']
#
# # loop through the subplots and plot each column
# for i, ax in enumerate(axes.flat):
#     if i < len(columns_to_plot):
#         ax.boxplot(x=Weather_Data.data[columns_to_plot[i]])
#         ax.set_title(columns_to_plot[i])
#     else:
#         ax.set_visible(False)
#
# plt.tight_layout()  # adjust the spacing between subplots
# plt.show()  # display the plot
#
# # Check if the outliers are removed!
# fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 8))
#
# # select the columns to plot
# columns_to_plot = ['pressure', 'wind_speed', 'rain_1h', 'rain_3h']
#
# # loop through the subplots and plot each column
# for i, ax in enumerate(axes.flat):
#     if i < len(columns_to_plot):
#         ax.boxplot(x=Weather_Data.data[columns_to_plot[i]])
#         ax.set_title(columns_to_plot[i])
#     else:
#         ax.set_visible(False)
#
# plt.tight_layout()  # adjust the spacing between subplots
# plt.show()  # display the plot
