import xgboost as xgb
from sklearn.metrics import mean_absolute_error
from DataHandler.Dataloader import DataReader
from sklearn.preprocessing import MinMaxScaler
from Visualization.Plot_Tool import DataMonitor
from Model.LSTM_Attention_xgboot import *

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

x_train_xgb = DataReader.reshaping(x_train)
x_val_xgb = DataReader.reshaping(x_val)
x_test_xgb = DataReader.reshaping(x_test)

input_unit = 256
hist_size = 24
activation = "tanh"
epoch = 100
batch_size = 64


model = LSTM_Attention(input_unit, hist_size, activation, x_train, y_train, (x_val, y_val), epoch, batch_size)
# model.plot_learning_rate()

model.compile_model(loss='mean_absolute_error')
model.summary_model()

# Model Training

print('')
print('LSTM-Attention is fitting a model on train-validation sets...')
print('')

history = model.train_model()

# Predictions and Inverse Transform

print('')
print('----------------------------------')
print('LSTM-attention is predicting...')
print('')

residuals_train = y_train - model.predict_model(x_train)
residuals_val = y_val - model.predict_model(x_val)

print('')
print('----------------------------------')
print('XGBoost is fitting a model on residuals...')
print('')


xgb_hybrid = xgb.XGBRegressor(
    n_estimators=1000,
    eval_metric='mae',
    early_stopping_rounds=8,
    verbosity=0
)

xgb_hybrid.fit(
    x_train_xgb,
    residuals_train,
    eval_set=[(x_train_xgb, residuals_train), (x_val_xgb, residuals_val)],
    verbose=False
)

y_train_pred = model.predict_model(x_train) + xgb_hybrid.predict(x_train_xgb).reshape(-1, 1)
y_val_pred = model.predict_model(x_val) + xgb_hybrid.predict(x_val_xgb).reshape(-1, 1)

y_test_pred = model.predict_model(x_test) + xgb_hybrid.predict(x_test_xgb).reshape(-1, 1)
print('')
print('')

print('---------------------------------------------------')
print(f'LSTM-Attention-XGBoost MAE for test set : {round(mean_absolute_error(y_test_pred,y_test),3)}')
print('---------------------------------------------------')
y_pred_actual = scaler_y.inverse_transform(y_test_pred)
print('')


# Start from here
y_test_inv = scaler_y.inverse_transform(y_test)


# Plotting function for Results

# Call the function to plot the results
datamonitor.plot_results(y_pred_actual, y_test_inv, history, 'LSTM-Attention-XGBoost',
                         "Prediction vs Actual Price for 1000 Observations", 'Observation', 'Phycoprotein')
