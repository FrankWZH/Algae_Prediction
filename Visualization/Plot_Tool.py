import seaborn as sns
import matplotlib.pyplot as plt
import plotly.graph_objects as go

from plotly.subplots import make_subplots
from statsmodels.graphics.tsaplots import plot_acf


class DataMonitor:

    def __init__(self, data, nrows=1, ncols=1, figsize=(14, 6)):
        self.data = data
        # self.fig = make_subplots()
        # self.fig, self.axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
        # self.plt = plt.figure(figsize=figsize)
        self.sns = sns

    @staticmethod
    def make_figure(figsize=(14, 6)):
        plt.figure(figsize=figsize)

    def make_subplots(self):
        self.fig = make_subplots()

    def add_trace(self, x, y, name):
        self.fig.add_trace(go.Line(x=x, y=y, name=name))

    @staticmethod
    def set_legend():
        plt.legend()

    def fig_show(self):
        self.fig.show()

    @staticmethod
    def plot_show():
        plt.show()

    def set_style(self, style="darkgrid"):
        self.sns.set(style=style)

    def add_barplot(self, xname: str, yname: str, ax_idx: int, title: str):
        self.sns.barplot(x=xname, y=yname, data=self.data, estimator=sum, color='royalblue', ax=self.axes[ax_idx])
        self.axes[ax_idx].set_title(title)

    def add_histplot(self, data_idx, label: str, bins=100, element="step", color='lightcoral', kde=True):
        return self.sns.histplot(self.data[data_idx], label=label, bins=bins, element=element, color=color, kde=kde)

    def add_plot(self, data, index: int, color: str, title: str, set_title=True, have_ax_idx=True):
        if have_ax_idx:
            self.axes[index].plot(data, color=color)
        else:
            self.axes.plot(data, color=color)
        if set_title:
            self.axes[index].set_title(title)

    def add_axvline(self, x, ax_idx: int, color: str, linestyle: str, have_ax_idx=True):
        if have_ax_idx:
            self.axes[ax_idx].axvline(x, color=color, linestyle=linestyle)
        else:
            self.axes.axvline(x, color=color, linestyle=linestyle)

    def set_title(self, ax_idx: int, title: str, have_ax_idx=True):
        if have_ax_idx:
            self.axes[ax_idx].set_title(title)
        else:
            self.axes.set_title(title)

    def add_plot_acf(self, data_idx: str, ax_idx: int, color: str, title: str):
        plot_acf(self.data[data_idx], lags=50, ax=self.axes[ax_idx], color=color)
        self.axes[ax_idx].set_title(title)

    def update_subplot(self, nrows=1, ncols=1, figsize=(14, 6)):
        self.fig, self.axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)

    def plot_results(self, y_pred_actual, y_test_inv, history, model_name, title, xlabel, ylabel):
        self.update_subplot(2, 1, (14, 6))

        # Plot Prediction vs Actual for the first 1000 observations
        self.axes[0].plot(y_pred_actual[:1000], label='Prediction')
        self.axes[0].plot(y_test_inv[:1000], label='Actual')
        self.axes[0].legend(loc='upper left')
        self.axes[0].set_title(title + f' ({model_name})')
        self.axes[0].set_xlabel(xlabel)
        self.axes[0].set_ylabel(ylabel)

        # Plot Training and Validation Loss
        self.axes[1].plot(history.history['loss'], label='Training Loss')
        self.axes[1].plot(history.history['val_loss'], label='Validation Loss')
        self.axes[1].legend()
        self.axes[1].set_title(f'Training and Validation Loss ({model_name})')
        self.axes[1].set_xlabel('Epochs')
        self.axes[1].set_ylabel('Loss')
        self.set_tight_layout()
        self.plot_show()

    @staticmethod
    def set_tight_layout():
        plt.tight_layout()
