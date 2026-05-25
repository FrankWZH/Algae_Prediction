import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import numpy as np

datapath1 = "/Users/wuzihang/Library/Containers/com.tencent.xinWeChat/Data/Library/Application Support/com.tencent.xinWeChat/2.0b4.0.9/b8d7b5a712dc3aee2ec27b9bc0a38a65/Message/MessageTemp/8da8b40b5476034d6d70cd485fcf38f8/File/11份数据集2.0/1监测数据.csv"
datapath2 = "/Users/wuzihang/Library/Containers/com.tencent.xinWeChat/Data/Library/Application Support/com.tencent.xinWeChat/2.0b4.0.9/b8d7b5a712dc3aee2ec27b9bc0a38a65/Message/MessageTemp/8da8b40b5476034d6d70cd485fcf38f8/File/11份数据集2.0/2监测数据.csv"
datapath3 = "/Users/wuzihang/Library/Containers/com.tencent.xinWeChat/Data/Library/Application Support/com.tencent.xinWeChat/2.0b4.0.9/b8d7b5a712dc3aee2ec27b9bc0a38a65/Message/MessageTemp/8da8b40b5476034d6d70cd485fcf38f8/File/11份数据集2.0/3监测数据.csv"
datapath4 = "/Users/wuzihang/Library/Containers/com.tencent.xinWeChat/Data/Library/Application Support/com.tencent.xinWeChat/2.0b4.0.9/b8d7b5a712dc3aee2ec27b9bc0a38a65/Message/MessageTemp/8da8b40b5476034d6d70cd485fcf38f8/File/11份数据集2.0/4监测数据.csv"
datapath5 = "/Users/wuzihang/Library/Containers/com.tencent.xinWeChat/Data/Library/Application Support/com.tencent.xinWeChat/2.0b4.0.9/b8d7b5a712dc3aee2ec27b9bc0a38a65/Message/MessageTemp/8da8b40b5476034d6d70cd485fcf38f8/File/11份数据集2.0/5监测数据.csv"
datapath6 = "/Users/wuzihang/Library/Containers/com.tencent.xinWeChat/Data/Library/Application Support/com.tencent.xinWeChat/2.0b4.0.9/b8d7b5a712dc3aee2ec27b9bc0a38a65/Message/MessageTemp/8da8b40b5476034d6d70cd485fcf38f8/File/11份数据集2.0/6监测数据.csv"
datapath7 = "/Users/wuzihang/Library/Containers/com.tencent.xinWeChat/Data/Library/Application Support/com.tencent.xinWeChat/2.0b4.0.9/b8d7b5a712dc3aee2ec27b9bc0a38a65/Message/MessageTemp/8da8b40b5476034d6d70cd485fcf38f8/File/11份数据集2.0/7监测数据.csv"
datapath8 = "/Users/wuzihang/Library/Containers/com.tencent.xinWeChat/Data/Library/Application Support/com.tencent.xinWeChat/2.0b4.0.9/b8d7b5a712dc3aee2ec27b9bc0a38a65/Message/MessageTemp/8da8b40b5476034d6d70cd485fcf38f8/File/11份数据集2.0/8监测数据.csv"
datapath9 = "/Users/wuzihang/Library/Containers/com.tencent.xinWeChat/Data/Library/Application Support/com.tencent.xinWeChat/2.0b4.0.9/b8d7b5a712dc3aee2ec27b9bc0a38a65/Message/MessageTemp/8da8b40b5476034d6d70cd485fcf38f8/File/11份数据集2.0/监测数据-半月湾6.12日喷洒-20240621183327.csv"
datapath10 = "/Users/wuzihang/Library/Containers/com.tencent.xinWeChat/Data/Library/Application Support/com.tencent.xinWeChat/2.0b4.0.9/b8d7b5a712dc3aee2ec27b9bc0a38a65/Message/MessageTemp/8da8b40b5476034d6d70cd485fcf38f8/File/11份数据集2.0/监测数据-独墅湖半月湾6.4-20240621183429.csv"
datapath11 = "/Users/wuzihang/Library/Containers/com.tencent.xinWeChat/Data/Library/Application Support/com.tencent.xinWeChat/2.0b4.0.9/b8d7b5a712dc3aee2ec27b9bc0a38a65/Message/MessageTemp/8da8b40b5476034d6d70cd485fcf38f8/File/11份数据集2.0/监测数据-独墅湖半月湾6.11-20240621183353.csv"


data1 = pd.read_csv(datapath1)
data2 = pd.read_csv(datapath2)
data3 = pd.read_csv(datapath3)
data4 = pd.read_csv(datapath4)
data5 = pd.read_csv(datapath5)
data6 = pd.read_csv(datapath6)
data7 = pd.read_csv(datapath7)
data8 = pd.read_csv(datapath8)
data9 = pd.read_csv(datapath9)
data10 = pd.read_csv(datapath10)
data11 = pd.read_csv(datapath11)


data_total = pd.concat([data1, data2, data3, data4, data5, data6, data7, data8, data9, data10, data11], axis=0)
data_total.to_csv("/Users/wuzihang/Desktop/bz.csv", index=False)
print(data_total)

data_dropped0 = data11.dropna()
data_dropped1 = data_dropped0.drop('Phycoprotein', axis=1)
data_dropped = data_dropped1.drop('Ammonia Nitrogen', axis=1)
print(data_dropped)

# 数据标准化
scaler = StandardScaler()
feature = scaler.fit_transform(data_dropped)
print(feature)

# 数据pca降维
pca = PCA(n_components=0.95)
reduced = pca.fit_transform(feature)
print(reduced)
print(pca.n_components)
ratio = pca.explained_variance_ratio_

print("方差解释比例：")
print(ratio)
print(type(ratio))

# 获取特征向量（主成分的线性组合权重）
components = pca.components_
print(type(components))

# 设置所有列/行不省略显示
pd.set_option('display.max_columns', None)  # 显示所有列
pd.set_option('display.max_rows', None)     # 显示所有行
pd.set_option('display.width', None)        # 自动调整宽度
pd.set_option('display.max_colwidth', None) # 列内容不截断

# 转换为 DataFrame，便于查看
features = ['Chlorophyll-a', 'Conductivity', 'DO', 'Turbidity', 'Temperature', 'pH']  # 特征名称
components_df0 = pd.DataFrame(components, columns=features,
                             index=[f'PC {i+1}' for i in range(components.shape[0])])

components_df = np.abs(components_df0)
print("每个主成分中每个特征的线性组合权重：")
print(components_df)
print(ratio)

'''
# 可视化每个主成分的特征权重
components_df.T.plot(kind='bar', figsize=(8, 6))
# plt.title('Feature Contributions to Principal Components')
plt.ylabel('Weight')
plt.xlabel('Features')
plt.xticks(rotation=0)
plt.legend(title='Principal Components')
plt.grid()
plt.show()
'''
'''
# 获取当前轴
ax = plt.gca()

# 绘制条形图
components_df.T.plot(kind='bar', figsize=(8, 6), ax=ax)

# 在每个条形上添加数值标签
for container in ax.containers:
    ax.bar_label(container, fmt='%.4f', padding=3)  # 显示2位小数，padding控制标签与条形的距离

plt.ylabel('Weight')
plt.xlabel('Features')
plt.xticks(rotation=0)
plt.legend(title='Principal Components')
plt.grid()
plt.tight_layout()  # 自动调整布局，防止标签被截断
plt.show()
'''

# 获取当前轴
ax = plt.gca()
# 设置全局字体大小
plt.rcParams.update({'font.size': 14})  # 调整全局基础字体大小
# 绘制条形图（通过width参数调整条形宽度）
components_df.T.plot(kind='bar', figsize=(10, 7), ax=ax, width=0.8)  # width默认为0.8，可增大到1.0或更大

# 在每个条形上添加数值标签（通过fontsize参数调整标签字体）
for container in ax.containers:
    ax.bar_label(container,
                fmt='%.4f',
                padding=3,
                fontsize=12,
                rotation=45)  # 标签旋转45度

# 调整坐标轴标签字体大小
ax.set_ylabel('PCL', fontsize=16)
ax.set_xlabel('Features', fontsize=16)
# 调整图例字体大小
plt.legend(title='PCs', title_fontsize=16, fontsize=14)
# 调整坐标轴刻度标签字体大小
ax.tick_params(axis='both', which='major', labelsize=14)
plt.xticks(rotation=0)
plt.grid()
plt.tight_layout()
plt.show()








'''
# T转置

absolute_component_tran = np.abs(components.T)
total = np.sum(ratio)
ratio_tran = ratio.T /total
final = (absolute_component_tran)@(ratio_tran)
visual_final = pd.DataFrame(final)
print(final)

visual_final.plot(kind='bar', figsize=(8, 6))
plt.title('Comprehensive Feature Contributions to Principal Components')
x = np.arange(6)  # x 是从 0 到 len(y)-1 的整数索引
plt.xticks(x, ['Chlorophyll-a', 'Conductivity', 'Dissolved Oxygen', 'Turbidity', 'Temperature', 'pH'])

# y坐标是每个feature的不同线性权重 与 每个component的方差解释比例
plt.ylabel('Scalar coefficient * Explained Variance Ratio')
plt.xlabel('Features')
plt.xticks(rotation=0)
plt.grid()
plt.show()
'''

# 你的原始计算部分
absolute_component_tran = np.abs(components.T)
total = np.sum(ratio)
ratio_tran = ratio.T / total
final = (absolute_component_tran) @ (ratio_tran)
visual_final = pd.DataFrame(final)
print(final)

'''
# 绘制条形图
ax = visual_final.plot(kind='bar', figsize=(8, 6))

# 在每个条形顶部添加数值标签（保留2位小数）
for container in ax.containers:
    ax.bar_label(container, fmt='%.4f', padding=3)  # padding控制标签与条形的距离

# plt.title('Comprehensive Feature Contributions to Principal Components')
plt.xticks(range(6), ['Chlorophyll-a', 'Conductivity', 'DO', 'Turbidity', 'Temperature', 'pH'], rotation=0)
plt.ylabel('Scalar coefficient * Explained Variance Ratio')
plt.xlabel('Features')
plt.legend().remove()
plt.grid()
plt.tight_layout()  # 防止标签被截断
plt.show()
'''

# 绘制条形图
ax = visual_final.plot(kind='bar', figsize=(10, 7))  # 稍微增大图形尺寸以适应更大的字体

# 在每个条形顶部添加数值标签（字体14，45度倾斜）
for container in ax.containers:
    ax.bar_label(container,
                fmt='%.4f',
                padding=3,
                fontsize=14      # 标签字体大小14
                     # 45度倾斜
                )     # 垂直对齐方式

# 设置x轴刻度标签（字体14）
plt.xticks(range(6),
          ['Chlorophyll-a', 'Conductivity', 'DO', 'Turbidity', 'Temperature', 'pH'],
          rotation=0,
          fontsize=12)  # x轴刻度字体14

# 设置y轴刻度标签（字体14）
plt.yticks(fontsize=14)

# 设置坐标轴标签（字体16）
plt.ylabel('Scalar coefficient * Explained Variance Ratio', fontsize=16)
plt.xlabel('Features', fontsize=16)

plt.legend().remove()
plt.grid()

# 调整布局防止标签被截断
plt.tight_layout()
plt.show()