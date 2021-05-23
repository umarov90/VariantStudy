from tensorflow.keras.layers import LeakyReLU, LayerNormalization, MultiHeadAttention,\
                                    Add, Embedding, Layer, Reshape, Dropout,\
                                    Dense, Conv1D, Input, Flatten, Activation, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
import tensorflow as tf

projection_dim = 128
num_heads = 8
transformer_units = [
    projection_dim * 2,
    projection_dim,
]
transformer_layers = 8


def simple_model(input_size, num_regions, cell_num):
    input_shape = (input_size, 4)
    inputs = Input(shape=input_shape)
    x = inputs
    x = Dropout(0.3, input_shape=(None, input_size, 4))(x)
    x = resnet_v2(x, 9, 2)
    num_patches = 783
    x = Dropout(0.3, input_shape=(None, num_patches, 2076))(x)

    # Encode patches.
    encoded_patches = PatchEncoder(num_patches, projection_dim)(x)

    # Create multiple layers of the Transformer block.
    for _ in range(transformer_layers):
        # Layer normalization 1.
        x1 = LayerNormalization(epsilon=1e-6)(encoded_patches)
        # Create a multi-head attention layer.
        attention_output = MultiHeadAttention(
            num_heads=num_heads, key_dim=projection_dim, dropout=0.1
        )(x1, x1)
        # Skip connection 1.
        x2 = Add()([attention_output, encoded_patches])
        # Layer normalization 2.
        x3 = LayerNormalization(epsilon=1e-6)(x2)
        # MLP.
        x3 = mlp(x3, hidden_units=transformer_units, dropout_rate=0.1)
        # Skip connection 2.
        encoded_patches = Add()([x3, x2])

    # Create a [batch_size, projection_dim] tensor.
    representation = LayerNormalization(epsilon=1e-6)(encoded_patches)
    representation = Flatten()(representation)
    representation = Dropout(0.5)(representation)

    x = Dense(cell_num * num_regions)(representation)
    x = LeakyReLU(alpha=0.2)(x)
    outputs = Reshape((cell_num, num_regions))(x)
    model = Model(inputs, outputs, name="model")
    return model


def resnet_layer(inputs,
                 num_filters=16,
                 kernel_size=3,
                 strides=1,
                 activation='relu',
                 batch_normalization=True,
                 conv_first=True):

    conv = Conv1D(num_filters,
                  kernel_size=kernel_size,
                  strides=strides,
                  padding='same',
                  # kernel_regularizer=l2(1e-6),
                  # activity_regularizer=l2(1e-6)
                  )

    x = inputs
    if conv_first:
        x = conv(x)
        if batch_normalization:
            x = BatchNormalization()(x)
        if activation is not None:
            x = Activation(activation)(x)
    else:
        if batch_normalization:
            x = BatchNormalization()(x)
        if activation is not None:
            x = Activation(activation)(x)
        x = conv(x)
    return x


def resnet_v2(input_x, num_stages, num_res_blocks):
    # Start model definition.
    num_filters_in = 16

    # v2 performs Conv2D with BN-ReLU on input before splitting into 2 paths
    x = resnet_layer(inputs=input_x,
                     num_filters=num_filters_in,
                     conv_first=True)

    # Instantiate the stack of residual units
    for stage in range(num_stages):
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
                num_filters_out = int(num_filters_in * 1.3) # changed from 2
                if res_block == 0:  # first layer but not first stage
                    strides = 2    # downsample

            # bottleneck residual unit
            y = resnet_layer(inputs=x,
                             num_filters=num_filters_in,
                             kernel_size=1,
                             strides=strides,
                             activation=activation,
                             batch_normalization=batch_normalization,
                             conv_first=False)
            y = resnet_layer(inputs=y,
                             num_filters=num_filters_in,
                             conv_first=False)
            y = resnet_layer(inputs=y,
                             num_filters=num_filters_out,
                             kernel_size=1,
                             conv_first=False)
            if res_block == 0:
                # linear projection residual shortcut connection to match
                # changed dims
                x = resnet_layer(inputs=x,
                                 num_filters=num_filters_out,
                                 kernel_size=1,
                                 strides=strides,
                                 activation=None,
                                 batch_normalization=False)
            x = Add()([x, y])

        num_filters_in = num_filters_out

    # Add classifier on top.
    # v2 has BN-ReLU before Pooling
    x = LayerNormalization()(x)
    x = Activation('relu')(x)
    # x = AveragePooling1D(pool_size=8)(x)

    return x


def mlp(x, hidden_units, dropout_rate):
    for units in hidden_units:
        x = Dense(units, activation=tf.nn.gelu)(x)
        x = Dropout(dropout_rate)(x)
    return x


class PatchEncoder(Layer):
    def __init__(self, num_patches, projection_dim, **kwargs):
        super(PatchEncoder, self).__init__(**kwargs)
        self.num_patches = num_patches
        self.projection = Dense(units=projection_dim)
        self.projection_dim = projection_dim
        self.position_embedding = Embedding(
            input_dim=num_patches, output_dim=projection_dim
        )

    def call(self, patch):
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        encoded = self.projection(patch) + self.position_embedding(positions)
        return encoded

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'num_patches': self.num_patches,
            'projection_dim': self.projection_dim,
        })
        return config