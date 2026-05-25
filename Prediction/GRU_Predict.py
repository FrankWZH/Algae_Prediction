from sklearn.metrics import mean_absolute_error
from DataHandler.Dataloader import DataReader
from sklearn.preprocessing import MinMaxScaler
from Visualization.Plot_Tool import DataMonitor
from Model.GRU import *
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


col_names = ["Ammonia Nitrogen", "Turbidity", "EC"]
col_names2 = ["Ammonia Nitrogen"]

algae_data = DataReader("../bz.csv", col_names)
algae_data.interpolate_nan()
y_algae_data = algae_data.data[["Phycoprotein"]].copy(deep=True)
algae_data.drop_column(['Phycoprotein'])
x_algae_data = algae_data.data.copy(deep=True)
print(x_algae_data)
print("haha")

'''
params_pca = {'cum_variance': 0.8, 'if_apply': True}
x_algae_data = DataReader.apply_pca(x_algae_data, **params_pca)


train_cutoff = int(0.8*x_algae_data.shape[0])
val_cutoff = int(0.9*x_algae_data.shape[0])

scaler_y = MinMaxScaler()
scaler_y.fit(y_algae_data[:train_cutoff])
y_norm = scaler_y.transform(y_algae_data)

hist_size = 24

data_norm = np.concatenate((x_algae_data, y_norm), axis=1)
print(data_norm)
print("hha")
'''

data_norm = np.concatenate((x_algae_data, y_algae_data), axis=1)
print(data_norm)
hist_size = 24

# 1. 初始数据划分（保持原始代码）
train_cutoff = int(0.8 * data_norm.shape[0])  # 前80%训练集
val_cutoff = int(0.9 * data_norm.shape[0])    # 80%~90%验证集，90%~100%测试集




# Train set, validation set, test set
x_train, y_train = DataReader.windowing(data_norm[:train_cutoff, :], data_norm[:train_cutoff, -1], hist_size)
x_val, y_val = DataReader.windowing(data_norm[train_cutoff: val_cutoff, :], data_norm[train_cutoff:val_cutoff, -1],
                                    hist_size)
x_test, y_test = DataReader.windowing(data_norm[val_cutoff:, :], data_norm[val_cutoff:, -1], hist_size)



# 3. 标准化处理（按特征维度标准化）
# 注意：x_train/x_val/x_test是3D张量（样本数, 时间步长, 特征数），需要先展平为2D标准化
def flatten_for_scaling(x):
    return x.reshape(-1, x.shape[-1])  # 合并样本和时间步，保留特征维度

scaler = StandardScaler()
# 只在训练集上拟合标准化器
scaler.fit(flatten_for_scaling(x_train))

# 对全部数据集应用标准化
x_train_scaled = scaler.transform(flatten_for_scaling(x_train)).reshape(x_train.shape)
x_val_scaled = scaler.transform(flatten_for_scaling(x_val)).reshape(x_val.shape)
x_test_scaled = scaler.transform(flatten_for_scaling(x_test)).reshape(x_test.shape)

# 4. PCA降维（可选）
params_pca = {'n_components': 0.8}  # 保留80%方差
pca = PCA(**params_pca)

# 只在训练集上拟合PCA
pca.fit(flatten_for_scaling(x_train_scaled))

# 应用PCA到所有数据集
def apply_pca(x):
    original_shape = x.shape
    flattened = x.reshape(-1, original_shape[-1])
    pca_transformed = pca.transform(flattened)
    # 调整形状为 (样本数, 时间步长, PCA后的特征数)
    return pca_transformed.reshape(original_shape[0], original_shape[1], -1)

x_train = apply_pca(x_train_scaled)
print(x_train)
x_val = apply_pca(x_val_scaled)
x_test = apply_pca(x_test_scaled)

# 最终得到：
# x_train_pca: (n_train_samples, hist_size, n_pca_features)
# x_val_pca:   (n_val_samples, hist_size, n_pca_features)
# x_test_pca:  (n_test_samples, hist_size, n_pca_features)

scaler_y = StandardScaler()
scaler_y.fit(y_train)

y_train = scaler_y.transform(y_train)
y_val = scaler_y.transform(y_val)
y_test = scaler_y.transform(y_test)

print(y_train)
print(len(y_train))
print(y_test)
print(len(y_test))
print(y_val)
print(len(y_val))






algae_data = DataReader("../bz.csv", col_names)
algae_data.interpolate_nan()

datamonitor = DataMonitor(x_algae_data)
datamonitor.update_subplot()

datamonitor.add_plot(algae_data.data['Phycoprotein'].iloc[:train_cutoff], 0, 'black', '',
                     False, have_ax_idx=False)
datamonitor.add_plot(algae_data.data['Phycoprotein'].iloc[train_cutoff + 1: val_cutoff], 0, 'green',
                     '', False, have_ax_idx=False)
datamonitor.add_plot(algae_data.data['Phycoprotein'].iloc[val_cutoff + 1:], 0, 'blue', '',
                     False, have_ax_idx=False)
datamonitor.add_axvline(algae_data.data.index[train_cutoff], 0, 'green', '--',
                        have_ax_idx=False)
datamonitor.add_axvline(algae_data.data.index[val_cutoff], 0, 'blue', '--',
                        have_ax_idx=False)

datamonitor.axes.set_xlabel('Sample')
datamonitor.axes.set_ylabel('Phycoprotein')

datamonitor.set_title(0, 'Visual split of train (black), validation (green) and test (blue) sets',
                      have_ax_idx=False)

# datamonitor.plot_show()
datamonitor.plot_show()

input_unit = 256
activation = "tanh"
epoch = 100
batch_size = 100


model = GRU(input_unit, activation, x_train, y_train, (x_val, y_val), epoch, batch_size)
model.plot_learning_rate()

model.compile_model(loss='mean_absolute_error')
model.summary_model()

# Model Training
history = model.train_model()

# Predictions and Inverse Transform
y_pred = model.predict_model(x_test)
print('')
print('')
print('---------------------------------------------------')
print(f'GRU MAE for test set : {round(mean_absolute_error(y_pred,y_test), 6)}')
print('---------------------------------------------------')

y_pred_actual = scaler_y.inverse_transform(y_pred)
print('')

# Start from here
y_test_inv = scaler_y.inverse_transform(y_test)


# Plotting function for Results

# Call the function to plot the results
datamonitor.plot_results(y_pred_actual, y_test_inv, history, 'GRU',
                         "Prediction vs Actual Price for 1000 Observations", 'Observation', 'Phycoprotein')
