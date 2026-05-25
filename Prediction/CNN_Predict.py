from sklearn.metrics import mean_absolute_error
from DataHandler.Dataloader import DataReader
from sklearn.preprocessing import MinMaxScaler
from Visualization.Plot_Tool import DataMonitor
from Model.CNN import *

col_names = ["Ammonia Nitrogen", "Turbidity", "EC"]

algae_data = DataReader("../bz.csv", col_names)
algae_data.interpolate_nan()
y_algae_data = algae_data.data[["Phycoprotein"]].copy(deep=True)
algae_data.drop_column(['Phycoprotein'])
x_algae_data = algae_data.data.copy(deep=True)


params_pca = {'cum_variance': 0.8, 'if_apply': True}
x_algae_data = DataReader.apply_pca(x_algae_data, **params_pca)


train_cutoff = int(0.8*x_algae_data.shape[0])
val_cutoff = int(0.9*x_algae_data.shape[0])

scaler_y = MinMaxScaler()
scaler_y.fit(y_algae_data[:train_cutoff])
y_norm = scaler_y.transform(y_algae_data)

hist_size = 24
data_norm = np.concatenate((x_algae_data, y_norm), axis=1)

# Train set, validation set, test set
x_train, y_train = DataReader.windowing(data_norm[:train_cutoff, :], data_norm[:train_cutoff, -1], hist_size)
x_val, y_val = DataReader.windowing(data_norm[train_cutoff: val_cutoff, :], data_norm[train_cutoff:val_cutoff, -1],
                                    hist_size)
x_test, y_test = DataReader.windowing(data_norm[val_cutoff:, :], data_norm[val_cutoff:, -1], hist_size)

algae_data = DataReader("../bz.csv", col_names)
algae_data.interpolate_nan()

datamonitor = DataMonitor(x_algae_data)
# datamonitor.update_subplot()
#
# datamonitor.add_plot(algae_data.data['Phycoprotein'].iloc[:train_cutoff], 0, 'black', '',
#                      False, have_ax_idx=False)
# datamonitor.add_plot(algae_data.data['Phycoprotein'].iloc[train_cutoff + 1: val_cutoff], 0, 'green',
#                      '', False, have_ax_idx=False)
# datamonitor.add_plot(algae_data.data['Phycoprotein'].iloc[val_cutoff + 1:], 0, 'blue', '',
#                      False, have_ax_idx=False)
# datamonitor.add_axvline(algae_data.data.index[train_cutoff], 0, 'green', '--',
#                         have_ax_idx=False)
# datamonitor.add_axvline(algae_data.data.index[val_cutoff], 0, 'blue', '--',
#                         have_ax_idx=False)
#
# datamonitor.axes.set_xlabel('Sample')
# datamonitor.axes.set_ylabel('Phycoprotein')
#
# datamonitor.set_title(0, 'Visual split of train (black), validation (green) and test (blue) sets',
#                       have_ax_idx=False)
#
# datamonitor.plot_show()


filters = 128
kernel_size = 5
activation = "tanh"
epoch = 100
batch_size = 128

model = CNN(filters, kernel_size, activation, x_train, y_train, (x_val, y_val), epoch, batch_size)
# model.plot_learning_rate()

model.compile_model(loss='mean_absolute_error')
model.summary_model()

# Model Training
history = model.train_model()

# Predictions and Inverse Transform
y_pred = model.predict_model(x_test)
print('')
print('')
print('---------------------------------------------------')
print(f'CNN MAE for test set : {round(mean_absolute_error(y_pred, y_test), 3)}')
print('---------------------------------------------------')
y_pred_actual = scaler_y.inverse_transform(y_pred)
print('')

# Start from here
y_test_inv = scaler_y.inverse_transform(y_test)


# Plotting function for Results

# Call the function to plot the results
datamonitor.plot_results(y_pred_actual, y_test_inv, history, 'CNN',
                         "Prediction vs Actual Price for 1000 Observations", 'Observation', 'Phycoprotein')

