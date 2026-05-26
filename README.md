<div align="center">

# 🌊 Algae Prediction

### A Deep Learning Framework for Algal Bloom Forecasting

Built with **Python**, **TensorFlow/PyTorch**, **LSTM/GRU/CNN**, and **XGBoost**

<p align="center">
  <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white">
  <img src="https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white">
  <img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white">
  <img src="https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white">
  <img src="https://img.shields.io/badge/pandas-150458?style=for-the-badge&logo=pandas&logoColor=white">
  <img src="https://img.shields.io/badge/XGBoost-005571?style=for-the-badge">
  <img src="https://img.shields.io/badge/Status-Active-brightgreen?style=for-the-badge">
</p>

<p align="center">
  <b>LSTM • GRU • CNN • Attention Mechanism • CNN-LSTM • LSTM-Attention • XGBoost Integration</b>
</p>

</div>

---

# 📖 Overview

**Algae Prediction** is a deep learning framework designed for algal bloom forecasting using environmental time-series data.  
The project implements and compares multiple neural network architectures and ensemble learning approaches to improve prediction accuracy and forecasting robustness.

This repository focuses on:

- Environmental time-series forecasting
- Deep learning model experimentation
- Hybrid neural architectures
- Ensemble learning with XGBoost
- Scientific data visualization

---

# ✨ Key Features

## 🧠 Deep Learning Models

| Model | File | Description |
|-------|------|-------------|
| **LSTM** | `LSTM.py` | Long Short-Term Memory for sequential prediction |
| **GRU** | `GRU.py` | Lightweight recurrent neural network |
| **CNN** | `CNN.py` | Convolutional Neural Network for feature extraction |

---

## 🤝 Ensemble Models

| Model | File | Description |
|-------|------|-------------|
| **CNN-LSTM** | `CNN_LSTM.py` | Hybrid spatial-temporal architecture |
| **LSTM-Attention** | `LSTM_Attention.py` | Attention-enhanced sequence learning |
| **GRU-XGBoost** | `GRU_xgboot.py` | GRU feature extraction + XGBoost |
| **LSTM-Attention-XGBoost** | `LSTM_Attention_xgboot.py` | Attention model integrated with XGBoost |

---

# 🛠 Tech Stack

| Category | Technologies |
|----------|--------------|
| Language | Python 3.8+ |
| Deep Learning | TensorFlow / Keras, PyTorch |
| Machine Learning | scikit-learn, XGBoost |
| Data Processing | Pandas, NumPy |
| Visualization | Matplotlib |

---

# 📂 Project Structure

```text
Algae_Prediction/
├── .idea/                          # IDE configuration
├── DataHandler/                    # Data preprocessing utilities
├── Feature_Filtering/              # Feature selection & engineering
├── Model/
│   ├── CNN.py
│   ├── CNN_LSTM.py
│   ├── GRU.py
│   ├── GRU_xgboot.py
│   ├── LSTM.py
│   ├── LSTM_Attention.py
│   └── LSTM_Attention_xgboot.py
├── Prediction/                     # Prediction & inference
├── Visualization/                  # Data visualization
├── bz.csv                          # Environmental dataset
├── requirements.txt                # Dependencies
├── LICENSE                         # MIT License
└── README.md                       # Documentation
```

---

# ⚙️ Installation

## 1️⃣ Clone Repository

```bash
git clone https://github.com/FrankWZH/Algae_Prediction.git
cd Algae_Prediction
```

---

## 2️⃣ Create Virtual Environment

### macOS / Linux

```bash
python -m venv venv
source venv/bin/activate
```

### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

---

## 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 4️⃣ Verify Installation

```bash
python -c "import tensorflow, torch, pandas, sklearn; print('Setup complete!')"
```

---

# 📊 Usage Examples

## 🔹 Train LSTM Model

```python
from Model.LSTM import LSTM
from DataHandler.Dataloader import DataReader

algae_data = DataReader("bz.csv", ["Ammonia Nitrogen", "Turbidity", "EC"])
algae_data.interpolate_nan()

x_train, y_train = DataReader.windowing(
    algae_data.data.values,
    algae_data.data.values[:, -1],
    24
)

model = LSTM(
    input_unit=64,
    activation="tanh",
    x_data=x_train,
    y_data=y_train,
    valid_data=(x_train, y_train),
    epoch=100,
    batch_size=128
)

model.compile_model(loss='mean_absolute_error')
history = model.train_model()

predictions = model.predict_model(x_train)
```

---

# 📈 Model Comparison

| Model | Complexity | Training Speed | Best For |
|-------|-------------|----------------|----------|
| LSTM | Medium | Moderate | Long-term dependencies |
| GRU | Low | Fast | Efficient forecasting |
| CNN | Low | Fast | Local pattern extraction |
| CNN-LSTM | High | Slow | Spatiotemporal prediction |
| LSTM-Attention | High | Slow | Interpretability |
| Ensemble Models | Very High | Slowest | Maximum accuracy |

---

# 📊 Dataset Information

The included `bz.csv` dataset contains environmental indicators related to algae growth and water quality.

Typical features include:

| Feature | Description |
|---------|-------------|
| Temperature | Water temperature |
| pH | Acidity / alkalinity |
| Dissolved Oxygen | Oxygen concentration |
| Nitrogen | Nutrient concentration |
| Phosphorus | Nutrient concentration |
| Chlorophyll-a | Algae biomass indicator |

> You may replace the dataset with your own environmental data for experimentation and research.

---

# 🔬 Research Objectives

This project aims to:

- Explore deep learning approaches for environmental forecasting
- Compare sequential neural architectures
- Improve algae bloom prediction accuracy
- Investigate ensemble learning strategies
- Provide reusable research modules

---

# 🤝 Contributing

Contributions are welcome!

## Development Workflow

```bash
# Fork repository
# Create feature branch
git checkout -b feature/new-feature

# Commit changes
git commit -m "Add new feature"

# Push branch
git push origin feature/new-feature
```

Then create a Pull Request on GitHub.

---

# 👥 Team Members

| Name | GitHub | Role | Contributions |
|------|--------|------|---------------|
| Yihang Xu |  | Lead Developer | Pipeline integration, deep learning implementation, documentation |
| Zihang Wu | [@FrankWZH](https://github.com/FrankWZH) | Deep Learning Engineer | Model optimization, ensemble methods, visualization |
| Kefan Wu | [@VenterWu](https://github.com/VenterWu) | Deep Learning Engineer | Model optimization, ensemble methods, visualization |

---

# 📅 Roadmap

## ✅ Completed

- Deep learning model implementations
- Ensemble learning integration
- Data preprocessing pipeline
- Visualization utilities

---

## 🚧 Planned Features

- [ ] Transformer-based models
- [ ] Streamlit dashboard
- [ ] Hyperparameter optimization
- [ ] Docker deployment
- [ ] Real-time forecasting API

---

# ⚠️ Disclaimer

This repository is intended for:

- Academic research
- Educational use
- Experimental deep learning studies

It is **not intended for production environmental decision-making** without additional scientific validation and expert evaluation.

---

# 📜 License

Distributed under the **MIT License**.

See the full license here:

➡️ https://github.com/FrankWZH/Algae_Prediction/blob/main/LICENSE

---

# Contact

## GitHub Repository

https://github.com/FrankWZH/Algae_Prediction

## Issues & Discussions

Please open an issue if you encounter bugs or would like to suggest improvements.

---


<div align="center">

## ⭐ If you find this project useful, consider giving it a star!

### Made with ❤️ for Environmental Deep Learning Research

</div>
