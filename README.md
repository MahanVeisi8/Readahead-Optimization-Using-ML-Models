# README - OS Lab Project: Readahead Optimization

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1Mfqw14bvlVBR-PubdpuC32GQ8yDYIFFg?usp=sharing)
[![Python Versions](https://img.shields.io/badge/Python-3.6%20%7C%203.7%20%7C%203.8-blue)](https://www.python.org/downloads/)
[![Dependencies Status](https://img.shields.io/badge/Dependencies-up%20to%20date-brightgreen)](https://github.com/username/repository/blob/main/requirements.txt)

## Introduction

This project is part of the Operating Systems Lab course, focusing on optimizing the **Readahead** feature of the Linux Page Cache. The project involves collecting data from various benchmarks, processing this data, and applying machine learning models to classify different workload types. The ultimate goal is to optimize the Readahead mechanism under varying workloads, using models like Decision Trees, Random Forests, and Neural Networks.

## Table of Contents

- [Project Overview](#project-overview)
- [Data Collection](#data-collection)
- [Data Processing](#data-processing)
- [Model Implementation](#model-implementation)
  - [Decision Tree](#decision-tree)
  - [Neural Network](#neural-network)
  - [Random Forest](#random-forest)
- [Results and Discussion](#results-and-discussion)
- [Conclusion](#conclusion)

## Project Overview

The project is centered around optimizing the Readahead feature, a prefetching technique used by the operating system to load data into the page cache before it is explicitly requested. The challenge lies in determining the optimal Readahead size for varying workloads, which include different I/O operations simulated using benchmarks like **RocksDB**.

### Problem Definition

Readahead can significantly impact the performance of I/O operations, especially under heavy workloads. However, if not tuned correctly, it can lead to cache pollution or unnecessary memory usage, degrading the system's overall performance. The project aims to develop a model that dynamically adjusts the Readahead size based on workload characteristics, using machine learning techniques.

### Objectives

1. **Data Collection:** Gather data on various I/O operations using RocksDB benchmarks and Linux's LTTng tracing framework.
2. **Feature Engineering:** Process the collected data to extract relevant features.
3. **Model Training:** Implement and train different models (Decision Tree, Neural Network, Random Forest) to classify workload types and suggest optimal Readahead sizes.
4. **Performance Evaluation:** Compare the performance of the models and determine the best approach.

## Data Collection

The data collection process involved running various RocksDB benchmarks on a Linux system with LTTng (Linux Trace Toolkit Next Generation) enabled to trace kernel-level I/O operations. The benchmarks included:

- `readrandom`
- `readseq`
- `readreverse`
- `readrandomwriterandom`

These benchmarks simulate different types of I/O operations, allowing us to collect a diverse dataset. The collected data includes timestamps, inode numbers, and the number of transactions, which were later processed to extract meaningful features.

### Commands Used

To start a recording session and capture relevant kernel events:
```bash
lttng create rs1 --output=/my-kernel-trace
lttng enable-event --kernel writeback_dirty_page,writeback_mark_inode_dirty
lttng start
```

To run the benchmarks:
```bash
db_bench --benchmarks="readrandom" --duration=600
db_bench --benchmarks="readseq" --duration=600
```

To stop the recording and process the data:
```bash
lttng destroy
babeltrace2 /my-kernel-trace > data.txt
```

## Data Processing

The raw data from the LTTng traces was processed to extract relevant features such as the timestamp (`Second`), inode number (`Ino`), and the number of transactions (`Transactions`). These features were normalized using Z-Score normalization to prepare them for model training.

```python
from sklearn.preprocessing import StandardScaler

X = df_all[['Second', 'Ino', 'Transactions']]
scaler = StandardScaler()
X_normalized = scaler.fit_transform(X)
```

## Model Implementation

### Decision Tree

A Decision Tree classifier was implemented to classify the workload types based on the extracted features. The model was evaluated using cross-validation to ensure its robustness.

```python
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score

dt_model = DecisionTreeClassifier(random_state=42)
cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
dt_scores = cross_val_score(dt_model, X_normalized, y, cv=cv, scoring='accuracy')
```

**Results:**
- **Accuracy:** 95.24%
- **Confusion Matrix:** 
- **Classification Report:** (See below)

![Decision Tree](images/decision_tree.png)

### Neural Network

A Neural Network was also trained to classify the workload types. The model architecture included fully connected layers with dropout for regularization. The training was monitored using early stopping to prevent overfitting.

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

model = Sequential()
model.add(Dense(128, input_dim=input_dim, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(output_dim, activation='softmax'))
```

**Results:**
- **Accuracy:** 98.67%
- **Loss Curve:**
![NN Loss Curve](images/nn_loss.png)

### Random Forest

To further improve the model's performance, a Random Forest classifier was implemented. Feature importance was analyzed to reduce the dimensionality of the dataset, retaining only the most significant features.

```python
from sklearn.ensemble import RandomForestClassifier

rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
```

**Results:**
- **Accuracy:** 97.85%
- **Feature Importance:**
![Feature Importance](images/feature_importance.png)

## Results and Discussion

The Neural Network outperformed the other models, achieving the highest accuracy on the test set. The Random Forest model also showed strong performance, particularly in terms of feature importance analysis. The Decision Tree, while less accurate, provided useful insights into the data's structure.

### Performance Comparison
| Model            | Accuracy  | Notes                                       |
|------------------|-----------|---------------------------------------------|
| Decision Tree    | 95.24%    | Simple, interpretable, but lower accuracy   |
| Neural Network   | 98.67%    | High accuracy, risk of overfitting          |
| Random Forest    | 97.85%    | Good balance of performance and interpretability |

## Conclusion

Through this project, we developed and compared several models to optimize the Readahead feature under varying workloads. The Neural Network provided the best performance, but the Random Forest's interpretability makes it a strong candidate for further exploration. Future work could involve integrating these models into a real-time system to dynamically adjust Readahead sizes based on the current workload.

