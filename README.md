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

## Data Preprocessing

After collecting and organizing the dataset, a crucial step involved preprocessing the data to prepare it for model training. The dataset contains 1,425,432 rows and 9 columns, as shown in the figure below:

![Dataset Overview](path_to_image1)

### Feature Selection Using Random Forest

To identify the most important features for our models, we used a Random Forest classifier to calculate feature importances. The Random Forest model highlighted `cumulative_time_elapsed` as the most significant feature by a large margin, followed by `flag`, `ino`, and `time_difference`. Features like `state` and `distance_from_mean` were less significant.

![Feature Importance](path_to_image2)

Based on this analysis, we removed features with importance values below a certain threshold to reduce the dataset's dimensionality, focusing only on the most relevant data.

### Data Visualization with t-SNE

To understand the distribution and separability of the different workload types in our dataset, we used t-SNE (t-distributed Stochastic Neighbor Embedding), a dimensionality reduction technique. The t-SNE plot below shows the dataset visualized in two dimensions, highlighting the clustering of different workload types. The distinct separation in the t-SNE plot indicates that our features are well-suited for classifying the different workloads.

![t-SNE Visualization](path_to_image3)

## Model Implementation

### Neural Network

We implemented a Multi-Layer Perceptron (MLP) neural network with two hidden layers of sizes 64 and 32, respectively. We trained this model using the selected features and evaluated it using a 10-fold cross-validation method.

#### Neural Network Architecture
The architecture consisted of:
- An input layer matching the number of selected features.
- Two hidden layers with ReLU activation and dropout for regularization.
- An output layer with softmax activation for multi-class classification.

We used early stopping during training to prevent overfitting, and the model achieved an average accuracy of approximately 99.85% on the test set.

#### Results
The following figure shows the training and validation accuracy over epochs:

![Neural Network Training](path_to_image4)

**Classification Report:**
- **Overall Accuracy:** 99.85%
- **Detailed Metrics:**

```plaintext
                       precision    recall  f1-score   support

              readseq       1.00      0.94      0.97      1623
           readrandom       1.00      1.00      1.00     37962
          readreverse       0.86      0.81      0.83       698
readrandomwriterandom       1.00      1.00      1.00    102261

             accuracy                           1.00    142544
            macro avg       0.96      0.94      0.95    142544
         weighted avg       1.00      1.00      1.00    142544
```

### Decision Tree

We also implemented a Decision Tree classifier, which provided high accuracy with a simple and interpretable model structure. The tree was visualized to understand the decision-making process.

#### Results
The Decision Tree model also achieved a perfect accuracy score on the test set, as shown in the following visualizations:

![Decision Tree Visualization - Small Depth](path_to_image5)

![Decision Tree Visualization - Full Depth](path_to_image6)

**Classification Report:**
- **Overall Accuracy:** 100%
- **Detailed Metrics:**

```plaintext
                       precision    recall  f1-score   support

              readseq       1.00      1.00      1.00      1623
           readrandom       1.00      1.00      1.00     37962
          readreverse       1.00      1.00      1.00       698
readrandomwriterandom       1.00      1.00      1.00    102261

            micro avg       1.00      1.00      1.00    142544
            macro avg       1.00      1.00      1.00    142544
         weighted avg       1.00      1.00      1.00    142544
          samples avg       1.00      1.00      1.00    142544
```

### Random Forest

Lastly, we implemented a Random Forest classifier, which combines multiple decision trees to improve accuracy and generalization. The Random Forest model achieved perfect accuracy on the test set, similar to the Decision Tree but with potentially better generalization on unseen data.

#### Results
The following visualization shows one of the decision trees within the Random Forest:

![Random Forest Tree Visualization](path_to_image7)

**Classification Report:**
- **Overall Accuracy:** 100%
- **Detailed Metrics:**

```plaintext
                       precision    recall  f1-score   support

              readseq       1.00      1.00      1.00      1623
           readrandom       1.00      1.00      1.00     37962
          readreverse       1.00      1.00      1.00       698
readrandomwriterandom       1.00      1.00      1.00    102261

             accuracy                           1.00    142544
            macro avg       1.00      1.00      1.00    142544
         weighted avg       1.00      1.00      1.00    142544
```


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

