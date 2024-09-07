# -*- coding: utf-8 -*-
"""OS_LabFinal.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1Mfqw14bvlVBR-PubdpuC32GQ8yDYIFFg

#introduction
This project is part of the Operating Systems Lab course, focusing on optimizing the Readahead feature of the Linux Page Cache. The project involves collecting data from various benchmarks, processing this data, and applying machine learning models to classify different workload types. The ultimate goal is to optimize the Readahead mechanism under varying workloads, using models like Decision Trees, Random Forests, and Neural Networks.


# Problem Definition
Readahead can significantly impact the performance of I/O operations, especially under heavy workloads. However, if not tuned correctly, it can lead to cache pollution or unnecessary memory usage, degrading the system's overall performance. The project aims to develop a model that dynamically adjusts the Readahead size based on workload characteristics, using machine learning techniques.

Objectives
Data Collection: Gather data on various I/O operations using RocksDB benchmarks and Linux's LTTng tracing framework.
Feature Engineering: Process the collected data to extract relevant features.
Model Training: Implement and train different models (Decision Tree, Neural Network, Random Forest) to classify workload types and suggest optimal Readahead sizes.
Performance Evaluation: Compare the performance of the models and determine the best approach.
# data gathering
"""

# mount it
from google.colab import drive
drive.mount('/content/drive')

"""# Working with larger and bigger dataset"""


from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from mpl_toolkits.mplot3d import Axes3D
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
from sklearn import tree
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical, plot_model
import numpy as np
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.tree import plot_tree
from sklearn.ensemble import RandomForestClassifier

ls

cd drive/MyDrive/AI/OS_lab

ls

dataset = pd.read_csv('Final-Data.csv')

dataset

sample_fraction = 0.5
sampled_data = dataset.sample(frac=sample_fraction, random_state=42)
print("Shape of Sampled Data:", sampled_data.shape)
print("\nSampled Data Overview:")
print(sampled_data.head())

dataset = sampled_data

X = dataset.drop('workload_type', axis=1)
y = dataset['workload_type']

"""trying to find out which feature to delete using random forest classifire"""

model = RandomForestClassifier()
model.fit(X, y)

feature_importances = model.feature_importances_
feature_names = X.columns
feature_importance_dict = dict(zip(feature_names, feature_importances))

sorted_features = sorted(feature_importance_dict.items(), key=lambda x: x[1], reverse=True)

plt.barh(range(len(sorted_features)), [val[1] for val in sorted_features], align='center')
plt.yticks(range(len(sorted_features)), [val[0] for val in sorted_features])
plt.xlabel('Feature Importance')
plt.title('Random Forest Feature Importance')
plt.show()

"""removing non-important features"""

threshold = 0.05
indices_to_keep = [i for i, importance in enumerate(feature_importances) if importance > threshold]
selected_features = X.iloc[:, indices_to_keep]

X = selected_features
print("\nShape of Updated Features (X):", X.shape)
print("\nUpdated Features (X):")
print(X.head())
print("\nShape of Updated Features (X):", y.shape)
print("\nTarget Variable (y):")
print(y.head())

X

"""using T-SNE to visualize our data
once in 2-D 😆
"""

!pip install openTSNE

X_data = selected_features.values

y_data = y.values
n_components_2d = 2
tsne_2d = TSNE(n_components=n_components_2d, random_state=40, n_jobs=-1)
X_tsne_2d = tsne_2d.fit(X_data)

plt.scatter(X_tsne_2d[:, 0], X_tsne_2d[:, 1], c=y_data, cmap='viridis', alpha=0.5)
plt.title('t-SNE 2D Visualization')
plt.xlabel('t-SNE Component 1')
plt.ylabel('t-SNE Component 2')
plt.show()

index_to_class_mapping = {1: 'readseq', 2:'readrandom', 3: 'readreverse', 4: 'readrandomwriterandom'}

"""using MPL nn models"""

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

nn_model = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42)
nn_model.fit(X_train, y_train)

nn_accuracy = nn_model.score(X_test, y_test)
print(f"Neural Network Accuracy: {nn_accuracy:.4f}")

"""creating a nn model from scratch using the sequentials"""

input_dim = X_train.shape[1]
output_dim = len(label_encoder.classes_)

def create_nn_model(input_dim, output_dim):
    model = Sequential()
    model.add(Dense(128, input_dim=input_dim, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(output_dim, activation='softmax'))

    model.compile(loss='sparse_categorical_crossentropy', optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])
    return model

skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
nn_accuracies = []
for train_index, test_index in skf.split(selected_features, y_encoded):
    X_train_fold, X_test_fold = selected_features.iloc[train_index], selected_features.iloc[test_index]
    y_train_fold, y_test_fold = y_encoded[train_index], y_encoded[test_index]

    nn_model = create_nn_model(input_dim, output_dim)
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    history = nn_model.fit(X_train_fold, y_train_fold, epochs=20, batch_size=64,
                            validation_data=(X_test_fold, y_test_fold),
                            callbacks=[early_stopping], verbose=0)
    _, nn_accuracy = nn_model.evaluate(X_test_fold, y_test_fold, verbose=0)
    nn_accuracies.append(nn_accuracy)

average_nn_accuracy = sum(nn_accuracies) / len(nn_accuracies)
print(f"Average Neural Network Accuracy (10-fold CV): {average_nn_accuracy:.4f}")

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

y_onehot = to_categorical(y_encoded)

X_train, X_test, y_train, y_test = train_test_split(X, y_onehot, test_size=0.2, random_state=42)

def create_nn_model(input_dim, output_dim):
    model = Sequential()
    model.add(Dense(128, input_dim=input_dim, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(output_dim, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])
    return model

input_dim = X_train.shape[1]
output_dim = y_onehot.shape[1]
nn_model = create_nn_model(input_dim, output_dim)

plot_model(nn_model, to_file='neural_network_model.png', show_shapes=True, show_layer_names=True)

early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

history = nn_model.fit(X_train, y_train, epochs=20, batch_size=64, validation_data=(X_test, y_test), callbacks=[early_stopping], verbose=2)
nn_loss, nn_accuracy = nn_model.evaluate(X_test, y_test, verbose=0)
print(f"Neural Network Accuracy: {nn_accuracy:.4f}")
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

nn_predictions = nn_model.predict(X_test)

nn_predictions_indices = nn_predictions.argmax(axis=1)
y_test_indices = y_test.argmax(axis=1)

print("Neural Network Classification Report:")
print(classification_report(y_test_indices, nn_predictions_indices, target_names=index_to_class_mapping.values()))

"""
4455/4455 [==============================] - 10s 2ms/step
Neural Network Classification Report:
                       precision    recall  f1-score   support

              readseq       1.00      0.94      0.97      1623
           readrandom       1.00      1.00      1.00     37962
          readreverse       0.86      0.81      0.83       698
readrandomwriterandom       1.00      1.00      1.00    102261

             accuracy                           1.00    142544
            macro avg       0.96      0.94      0.95    142544
         weighted avg       1.00      1.00      1.00    142544
         """
"""Greattt resaults on NN models!! 🙂

let's try dt
"""

dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train, y_train)

dt_accuracy = dt_model.score(X_test, y_test)
print(f"Decision Tree Accuracy: {dt_accuracy:.4f}")

plt.figure(figsize=(20, 10))
plot_tree(dt_model, filled=True, feature_names=X.columns, class_names=index_to_class_mapping.values(), rounded=True, fontsize=10)
plt.show()

dt_predictions = dt_model.predict(X_test)

print("Decision Tree Classification Report:")
print(classification_report(y_test, dt_predictions, target_names=index_to_class_mapping.values()))

"""
Decision Tree Classification Report:
                       precision    recall  f1-score   support

              readseq       1.00      1.00      1.00      1623
           readrandom       1.00      1.00      1.00     37962
          readreverse       1.00      1.00      1.00       698
readrandomwriterandom       1.00      1.00      1.00    102261

            micro avg       1.00      1.00      1.00    142544
            macro avg       1.00      1.00      1.00    142544
         weighted avg       1.00      1.00      1.00    142544
          samples avg       1.00      1.00      1.00    142544
          """
"""**Awesome!** 🆙

***Let's use Random forrest too***
"""

rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
selected_tree = rf_model.estimators_[0]

plt.figure(figsize=(20, 10))
plot_tree(selected_tree, filled=True, feature_names=X.columns, class_names=index_to_class_mapping.values(), rounded=True, fontsize=10)
plt.show()

"""as you see there are lots of trees for making the decision. That's why it is called Random Forrest! **🙂**"""

from sklearn.metrics import classification_report

rf_accuracy = rf_model.score(X_test, y_test)
print(f"Random Forest Accuracy: {rf_accuracy:.4f}")
rf_predictions = rf_model.predict(X_test)
print("Random Forest Classification Report:")
print(classification_report(y_test.argmax(axis=1), rf_predictions.argmax(axis=1), target_names=index_to_class_mapping.values()))

"""
Random Forest Accuracy: 1.0000
Random Forest Classification Report:
                       precision    recall  f1-score   support

              readseq       1.00      1.00      1.00      1623
           readrandom       1.00      1.00      1.00     37962
          readreverse       1.00      1.00      1.00       698
readrandomwriterandom       1.00      1.00      1.00    102261

             accuracy                           1.00    142544
            macro avg       1.00      1.00      1.00    142544
         weighted avg       1.00      1.00      1.00    142544

"""

"""# Performance Comparison

| Model            | Accuracy  | Notes                                       |
|------------------|-----------|---------------------------------------------|
| Decision Tree    | 100.00%   | Simple, interpretable, perfect accuracy     |
| Neural Network   | 99.85%    | High accuracy, complex model with slight variability in precision |
| Random Forest    | 100.00%   | Combines multiple trees for perfect accuracy and generalization |"""

"""
# Conclusion

Through this project, we developed and compared three models—Decision Tree, Neural Network, and Random Forest—to optimize the Readahead feature under varying workloads. Both the Decision Tree and Random Forest models achieved perfect accuracy, demonstrating their strength in handling this classification task. The Neural Network, while slightly less accurate, offered flexibility in model design and captured complex relationships within the data. Given these results, the Random Forest model stands out for its combination of accuracy and interpretability, making it a strong candidate for real-time systems that require dynamic adjustment of Readahead sizes based on current workloads.
"""