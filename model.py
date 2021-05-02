import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from tensorflow.keras import regularizers
from tensorflow.keras import backend as K
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Reshape
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Activation
from tensorflow.keras import layers
from tensorflow import keras
from scipy import stats
import tensorflow as tf
import gc
import pickle
import numpy as np
import random
import shutil


def simple_model(input_size):
    input_shape = (input_size, 4)
    inputs = Input(shape=input_shape)
    x = inputs
    x = layers.Conv1D(22, 3, padding='same')(x)
    x = layers.Conv1D(44, 3, padding='same')(x)

    x = Flatten()(x)
    outputs = Dense(input_size)(x)
    model = Model(inputs, outputs, name="model")
    return model