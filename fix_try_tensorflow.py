#!/usr/bin/env python
# coding: utf-8

# In[1]:


#beberapa library
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import glob

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, train_test_split, StratifiedShuffleSplit, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, cohen_kappa_score
from sklearn.metrics import precision_recall_curve, classification_report, ConfusionMatrixDisplay, multilabel_confusion_matrix
from sklearn.utils import shuffle

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import layers
from tensorflow.keras.layers import Dense, LSTM, TimeDistributed, Conv1D, MaxPooling1D, BatchNormalization
from tensorflow.keras.layers import concatenate, Flatten, Dropout, Add, BatchNormalization
from tensorflow.keras.layers import Conv2D, AveragePooling2D, MaxPooling2D, Input, Bidirectional
from tensorflow.keras.models import Sequential, Model



# In[2]:


gpus = tf.config.list_physical_devices('GPU')
print(gpus)
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
if gpus:
    tf.config.set_visible_devices(gpus[0], 'GPU')

print("Test built: {}".format(tf.test.is_built_with_cuda()))
with tf.device("gpu:0"):
   print("tf.keras code in this scope will run on GPU")


# In[3]:


import numpy as np
import os

# Path folder data
data_dir = '/home/internship/fajrul/MIMIC/MIMIC-BP/dataset'

# Load subject list dari file txt
import ast

def load_subjects(file_path):
    with open(file_path, 'r') as f:
        content = f.read().strip()
        subjects = ast.literal_eval(content)  # parsing isi list dengan aman
    return subjects

train_subjects = load_subjects(os.path.join(data_dir, 'train_subjects.txt'))
val_subjects   = load_subjects(os.path.join(data_dir, 'val_subjects.txt'))
test_subjects  = load_subjects(os.path.join(data_dir, 'test_subjects.txt'))

print (train_subjects)
# Fungsi untuk load data berdasarkan daftar subject
def load_data(subjects, data_dir):
    X = []
    Y = []
    missing_data = []
    
    for subject in subjects:
        subject = subject.strip().lower()
        ppg_file = os.path.join(data_dir, f'{subject}_ppg.npy')
        label_file = os.path.join(data_dir, f'{subject}_labels.npy')
        
        try:
            ppg_data = np.load(ppg_file, allow_pickle=True)
            label_data = np.load(label_file, allow_pickle=True)
            
            # Validasi shape data
            if ppg_data.shape[0] == label_data.shape[0]:
                X.append(ppg_data)
                Y.append(label_data)
            else:
                print(f'Warning: Dimensi tidak match untuk subject {subject}')
                missing_data.append(subject)
        except (FileNotFoundError, IOError) as e:
            print(f'Warning: {str(e)}')
            missing_data.append(subject)
    
    if missing_data:
        print(f'\nTotal {len(missing_data)} subject dengan data tidak lengkap:')
        print(missing_data)
    
    return X, Y

# Load data untuk masing-masing set
x_train, y_train = load_data(train_subjects, data_dir)
x_val, y_val     = load_data(val_subjects, data_dir)
x_test, y_test   = load_data(test_subjects, data_dir)

print(len(x_train), len(y_train))

if not x_train or not y_train:
    raise ValueError("Data training kosong")
if not x_val or not y_val:
    raise ValueError("Data validasi kosong")
if not x_test or not y_test:
    raise ValueError("Data testing kosong")

try:
    x_train_array = np.stack(x_train)
    x_val_array = np.stack(x_val)
    x_test_array = np.stack(x_test)

    y_train_array = np.stack(y_train)
    y_val_array = np.stack(y_val)
    y_test_array = np.stack(y_test)
except ValueError as e:
    print(f"Error saat stacking array: {str(e)}")

    print("Dimensi data train:", [x.shape for x in x_train])
    print("Dimensi label train:", [y.shape for y in y_train])
    raise

print(f'Jumlah data train: {len(x_train)}')
print(f'Jumlah data val: {len(x_val)}')
print(f'Jumlah data test: {len(x_test)}')

import numpy as np

# Stack arrays
x_train_array = np.stack(x_train)
x_val_array = np.stack(x_val)
x_test_array = np.stack(x_test)

y_train_array = np.stack(y_train)
y_val_array = np.stack(y_val)
y_test_array = np.stack(y_test)

# Print shapes
print('x_train_array shape:', x_train_array.shape)
print('x_val_array shape:', x_val_array.shape)
print('x_test_array shape:', x_test_array.shape)

print('y_train_array shape:', y_train_array.shape)
print('y_val_array shape:', y_val_array.shape)
print('y_test_array shape:', y_test_array.shape)

# Reshape x arrays
x_train_reshaped = x_train_array.reshape(-1, x_train_array.shape[2])  # (33000, 3750)
x_val_reshaped = x_val_array.reshape(-1, x_val_array.shape[2])        # (195*30, 3750)
x_test_reshaped = x_test_array.reshape(-1, x_test_array.shape[2])     # (229*30, 3750)

# Reshape y arrays
y_train_reshaped = y_train_array.reshape(-1, y_train_array.shape[2])  # (33000, 2)
y_val_reshaped = y_val_array.reshape(-1, y_val_array.shape[2])        # (195*30, 2)
y_test_reshaped = y_test_array.reshape(-1, y_test_array.shape[2])     # (229*30, 2)

print("x_train:", x_train_reshaped.shape)
print("y_train:", y_train_reshaped.shape)

print("x_val:", x_val_reshaped.shape)
print("y_val:", y_val_reshaped.shape)

print("x_test:", x_test_reshaped.shape)
print("y_test:", y_test_reshaped.shape)


# In[4]:


x_train = np.expand_dims(x_train_reshaped, axis=1).astype(np.float32)
x_val = np.expand_dims(x_val_reshaped, axis=1).astype(np.float32)
x_test = np.expand_dims(x_test_reshaped, axis=1).astype(np.float32)

y_train = y_train_reshaped.astype(np.float32)
y_val = y_val_reshaped.astype(np.float32)
y_test = y_test_reshaped.astype(np.float32)

print("x_train shape:", x_train.shape)
print("y_train shape:", y_train.shape)

print("x_val shape:", x_val.shape)
print("y_val shape:", y_val.shape)

print("x_test shape:", x_test.shape)
print("y_test shape:", y_test.shape)


# In[5]:


x_train_type, y_train_type = type(x_train), type(y_train)
x_val_type, y_val_type = type(x_val), type(y_val)
x_test_type, y_test_type = type(x_test), type(y_test)

x_train_shape, y_train_shape = x_train.shape, y_train.shape
x_val_shape, y_val_shape = x_val.shape, y_val.shape
x_test_shape, y_test_shape = x_test.shape, y_test.shape

(x_train_type, y_train_type, x_train_shape, y_train_shape), 
(x_val_type, y_val_type, x_val_shape, y_val_shape), 
(x_test_type, y_test_type, x_test_shape, y_test_shape)


# In[6]:


# Transpose dari (batch, 1, 3750) → (batch, 3750, 1)
x_train = np.transpose(x_train, (0, 2, 1))
x_val   = np.transpose(x_val, (0, 2, 1))
x_test  = np.transpose(x_test, (0, 2, 1))

print("x_train shape:", x_train.shape)
print("y_train shape:", y_train.shape)

print("x_val shape:", x_val.shape)
print("y_val shape:", y_val.shape)

print("x_test shape:", x_test.shape)
print("y_test shape:", y_test.shape)


# In[7]:


from tensorflow.keras import layers, models

model = models.Sequential([
    layers.Conv1D(16, kernel_size=5, activation='relu', input_shape=(3750, 1)),
    layers.MaxPooling1D(pool_size=2),
    layers.Conv1D(32, kernel_size=5, activation='relu'),
    layers.MaxPooling1D(pool_size=2),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(2)  # Output: SBP dan DBP
])

print(model.summary())


# In[ ]:


from livelossplot import PlotLossesKeras
opt = Adam(lr=1e-5)
model.compile(optimizer=opt, loss='mse', metrics=['mae'])

history = model.fit(
    x_train, 
    y_train, 
    epochs=100, 
    batch_size=32, 
    validation_data=(x_val, y_val),
    callbacks=[PlotLossesKeras()]
)

