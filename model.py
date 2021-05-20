import os
from tensorflow.keras import regularizers
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Reshape
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Conv1D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Activation
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Add
from tensorflow.keras.regularizers import L2
from tensorflow.keras import layers

import tensorflow as tf
import numpy as np

# tf.compat.v1.disable_eager_execution()

projection_dim = 64
num_heads = 4
transformer_units = [
    projection_dim * 2,
    projection_dim,
]
transformer_layers = 4

def simple_model(input_size, num_regions, cell_num):
    input_shape = (input_size, 4)
    inputs = Input(shape=input_shape, name="input_dna_sequences")
    x = Dropout(0.3, input_shape=(None, input_size, 4), name="dropout_input")(inputs)
    x = resnet_v2(x, 11)
    num_patches = 785
    x = Dropout(0.3, input_shape=(None, num_patches, 912), name="dropout_resnet")(x)

    # Encode patches.
    encoded_patches = PatchEncoder(num_patches, projection_dim)(x)

    # Create multiple layers of the Transformer block.
    for tl in range(transformer_layers):
        # Layer normalization 1.
        x1 = layers.LayerNormalization(epsilon=1e-6, name="lanorm_1_" + str(tl))(encoded_patches)
        # Create a multi-head attention layer.
        attention_output = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=projection_dim, dropout=0.1, name="mha_" + str(tl)
        )(x1, x1)
        # Skip connection 1.
        x2 = layers.Add(name="add_1_tl_" + str(tl))([attention_output, encoded_patches])
        # Layer normalization 2.
        x3 = layers.LayerNormalization(epsilon=1e-6, name="lanorm_2_" + str(tl))(x2)
        # MLP.
        x3 = mlp(x3, hidden_units=transformer_units, dropout_rate=0.1, name="mlp_"+str(tl))
        # Skip connection 2.
        encoded_patches = layers.Add(name="add_2_tl_" + str(tl))([x3, x2])

    # Create a [batch_size, projection_dim] tensor.
    representation = layers.LayerNormalization(epsilon=1e-6, name="lanorm_3")(encoded_patches)
    representation = layers.Flatten(name="flatten_rep")(representation)
    representation = layers.Dropout(0.5, name="dropout_rep")(representation)

    x = Dense(cell_num * num_regions, name="dense_output")(representation)
    x = LeakyReLU(alpha=0.2, name="leaky_output")(x)
    outputs = Reshape((cell_num, num_regions), name="reshape_output")(x)
    model = Model(inputs=inputs, outputs=outputs, name="expression_model")
    return model


def resnet_layer(inputs,
                 num_filters=16,
                 kernel_size=3,
                 strides=1,
                 activation='relu',
                 batch_normalization=True,
                 conv_first=True,
                 name="_"):
    """2D Convolution-Batch Normalization-Activation stack builder

    # Arguments
        inputs (tensor): input tensor from input image or previous layer
        num_filters (int): Conv2D number of filters
        kernel_size (int): Conv2D square kernel dimensions
        strides (int): Conv2D square stride dimensions
        activation (string): activation name
        batch_normalization (bool): whether to include batch normalization
        conv_first (bool): conv-bn-activation (True) or
            bn-activation-conv (False)

    # Returns
        x (tensor): tensor as input to the next layer
    """
    conv = Conv1D(num_filters,
                  kernel_size=kernel_size,
                  strides=strides,
                  padding='same',
                  kernel_regularizer=L2(1e-6),
                  activity_regularizer=L2(1e-6),
                  name=name+"conv1d_1")

    x = inputs
    if conv_first:
        x = conv(x)
        if batch_normalization:
            x = BatchNormalization(name=name+"bn")(x)
        if activation is not None:
            x = Activation(activation, name=name+"activation")(x)
    else:
        if batch_normalization:
            x = BatchNormalization(name=name+"bn")(x)
        if activation is not None:
            x = Activation(activation, name=name+"activation")(x)
        x = conv(x)
    return x


def resnet_v2(input_x, depth):
    """ResNet Version 2 Model builder [b]

    Stacks of (1 x 1)-(3 x 3)-(1 x 1) BN-ReLU-Conv2D or also known as
    bottleneck layer
    First shortcut connection per layer is 1 x 1 Conv2D.
    Second and onwards shortcut connection is identity.
    At the beginning of each stage, the feature map size is halved (downsampled)
    by a convolutional layer with strides=2, while the number of filter maps is
    doubled. Within each stage, the layers have the same number filters and the
    same filter map sizes.
    Features maps sizes:
    conv1  : 32x32,  16
    stage 0: 32x32,  64
    stage 1: 16x16, 128
    stage 2:  8x8,  256

    # Arguments
        input_shape (tensor): shape of input image tensor
        depth (int): number of core convolutional layers
        num_classes (int): number of classes (CIFAR10 has 10)

    # Returns
        model (Model): Keras model instance
    """
    if (depth - 2) % 9 != 0:
        raise ValueError('depth should be 9n+2 (eg 56 or 110 in [b])')
    # Start model definition.
    num_filters_in = 16
    num_res_blocks = int((depth - 2) / 9)

    # v2 performs Conv2D with BN-ReLU on input before splitting into 2 paths
    x = resnet_layer(inputs=input_x,
                     num_filters=num_filters_in,
                     conv_first=True, name="first_rl_")

    # Instantiate the stack of residual units
    for stage in range(8):
        for res_block in range(num_res_blocks):
            activation = 'relu'
            batch_normalization = True
            strides = 1
            if stage == 0:
                num_filters_out = int(num_filters_in * 16) # changed from 4
                if res_block == 0:  # first layer and first stage
                    activation = None
                    batch_normalization = False
            else:
                num_filters_out = int(num_filters_in * 1.2) # changed from 2
                if res_block == 0:  # first layer but not first stage
                    strides = 2    # downsample
            name_id = str(stage) + "_" + str(res_block)
            # bottleneck residual unit
            y = resnet_layer(inputs=x,
                             num_filters=num_filters_in,
                             kernel_size=1,
                             strides=strides,
                             activation=activation,
                             batch_normalization=batch_normalization,
                             conv_first=False, name=name_id + "_second_rl_")
            y = resnet_layer(inputs=y,
                             num_filters=num_filters_in,
                             conv_first=False, name=name_id + "_third_rl_")
            y = resnet_layer(inputs=y,
                             num_filters=num_filters_out,
                             kernel_size=1,
                             conv_first=False, name=name_id + "_fourth_rl_")
            if res_block == 0:
                # linear projection residual shortcut connection to match
                # changed dims
                x = resnet_layer(inputs=x,
                                 num_filters=num_filters_out,
                                 kernel_size=1,
                                 strides=strides,
                                 activation=None,
                                 batch_normalization=False, name=name_id + "_fifths_rl_")
            x = Add(name="add_res_" + name_id)([x, y])

        num_filters_in = num_filters_out

    # Add classifier on top.
    # v2 has BN-ReLU before Pooling
    x = BatchNormalization(name="resnet_last_bn")(x)
    x = Activation('relu', name="resnet_last_activation")(x)
    # x = AveragePooling1D(pool_size=8)(x)

    return x


# def get_callbacks():
#     lr_scheduler = LearningRateScheduler(lr_schedule)
#     lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1),
#                                    cooldown=0,
#                                    patience=5,
#                                    min_lr=0.5e-6)
#     return [lr_reducer, lr_scheduler]


def lr_schedule(epoch):
    """Learning Rate Schedule

    Learning rate is scheduled to be reduced after 80, 120, 160, 180 epochs.
    Called automatically every epoch as part of callbacks during training.

    # Arguments
        epoch (int): The number of epochs

    # Returns
        lr (float32): learning rate
    """
    lr = 1e-3
    if epoch > 180:
        lr *= 0.5e-3
    elif epoch > 160:
        lr *= 1e-3
    elif epoch > 120:
        lr *= 1e-2
    elif epoch > 80:
        lr *= 1e-1
    print('Learning rate: ', lr)
    return lr


def mlp(x, hidden_units, dropout_rate, name):
    for u, units in enumerate(hidden_units):
        x = layers.Dense(units, activation=tf.nn.gelu, name=name + "_dense_" + str(u))(x)
        x = layers.Dropout(dropout_rate, name=name + "_dropout_" + str(u))(x)
    return x


class PatchEncoder(layers.Layer):
    def __init__(self, num_patches, projection_dim):
        super(PatchEncoder, self).__init__()
        self.num_patches = num_patches
        self.projection = layers.Dense(units=projection_dim)
        self.position_embedding = layers.Embedding(
            input_dim=num_patches, output_dim=projection_dim
        )

    def call(self, patch):
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        encoded = self.projection(patch) + self.position_embedding(positions)
        return encoded