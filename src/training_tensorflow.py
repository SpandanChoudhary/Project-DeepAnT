import os
os.chdir('D:\\Projects\\Project-DeepAnT')

# libraries
import warnings
import numpy as np
import pandas as pd
from utils import *
import keras_tuner as kt
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Conv1D, Flatten, Dense, Dropout, MaxPool1D

warnings.filterwarnings('ignore')

batch_sample_train = np.load('input/batch_sample_train.npy')
batch_sample_vali = np.load('input/batch_sample_vali.npy')

batch_label_train = np.load('input/batch_label_train.npy')
batch_label_vali = np.load('input/batch_label_vali.npy')

def build_model(hp):

    w = 20
    p_w = 1
    n_features = 1

    kernel_size = 2             # Size of filter in conv layers
    conv1D_1 = 32               # Number of filters in first conv layer
    conv1D_2 = 32               # Number of filters in second conv layer
    dense_1_nodes = 40          # Number of neurons in dense layer
    output_nodes = p_w          # Number of neurons in output layer

    hp_model = Sequential()

    # Hyper-parameter: Regularization Value
    hp_regular = hp.Choice('regularization_value', values=[0.1, 0.01, 0.001, 0.0001])

    # Conv1D Layer
    hp_model.add(Conv1D(filters=conv1D_1, kernel_size=kernel_size,
                        strides=1, padding='valid', activation='relu',
                        input_shape=(w, n_features),
                        kernel_regularizer=keras.regularizers.l1_l2(l1=hp_regular, l2=hp_regular)))

    # Pooling Layer #1
    hp_model.add(MaxPool1D(pool_size=2))

    # Conv1D layer
    hp_model.add(Conv1D(filters=conv1D_2, kernel_size=kernel_size,
                        strides=1, padding='valid', activation='relu',
                        kernel_regularizer=keras.regularizers.l1_l2(l1=hp_regular, l2=hp_regular)))

    # Max Pooling Layer #2
    hp_model.add(MaxPool1D(pool_size=2))

    # Flatten
    hp_model.add(Flatten())

    # Dense Layer
    hp_model.add(Dense(units=dense_1_nodes, activation='relu',
                       kernel_regularizer=keras.regularizers.l1_l2(l1=hp_regular, l2=hp_regular)))

    # Hyper-parameter: Dropout Rate
    hp_dropout = hp.Choice('dropout_rate', values=[0.2, 0.3, 0.4, 0.5])

    hp_model.add(Dropout(hp_dropout))

    hp_model.add(Dense(units=output_nodes,
                       kernel_regularizer=keras.regularizers.l1_l2(l1=hp_regular, l2=hp_regular)))

    # Model Compilation
    # Hyper-parameter: Learning Rate
    hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4, 1e-5, 1e-6])

    hp_model.compile(optimizer=keras.optimizers.Adam(lr=hp_learning_rate),
                     loss=keras.losses.MeanAbsoluteError())
    return hp_model

def model_training():
    
    # keras tuner 
    hp_tuner = kt.Hyperband(build_model, objective= 'val_loss', max_epochs=50, 
                        directory = 'tuner_directory', project_name = 'DeepAnT', overwrite = True)
    hp_tuner.search(batch_sample_train, batch_label_train,
                validation_data=(batch_sample_vali, batch_label_vali), shuffle = False)

    # Optimal hyperparameters
    best_hp = hp_tuner.get_best_hyperparameters(num_trials=1)[0]
    tuned_model = hp_tuner.hypermodel.build(best_hp)

    # tensorflow callbacks
    stop_early = keras.callbacks.EarlyStopping(monitor= 'val_loss', patience = 10, verbose= 1)
    best_model = keras.callbacks.ModelCheckpoint('models/deepant.h5', monitor='val_loss', 
                                             save_best_only= True, verbose = 1 )

    history = tuned_model.fit(batch_sample_train, batch_label_train, 
                          validation_data=(batch_sample_vali, batch_label_vali), 
                          epochs = 3000, callbacks = [stop_early, best_model])
    prediction_model = keras.models.load_model('models/deepant.h5')
    print(prediction_model.summary())

if __name__ =='__main__':

    model_training()