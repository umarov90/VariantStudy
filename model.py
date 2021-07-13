from tensorflow.keras.layers import LeakyReLU, LayerNormalization, MultiHeadAttention,\
                                    Add, Embedding, Layer, Reshape, Dropout,\
                                    Dense, Conv1D, Input, Flatten, Activation, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
import tensorflow as tf
import common as cm


projection_dim = 64
num_heads = 6
transformer_units = [
    projection_dim * 2,
    projection_dim,
]
transformer_layers = 4


def simple_model(input_size, num_regions, cell_num):
    input_shape = (input_size, 4)
    inputs = Input(shape=input_shape)
    x = inputs
    x = Dropout(0.2)(x)
    x = resnet_v2(x, 9, 2)
    num_patches = 79
    x = Dropout(0.5)(x)

    # # Encode patches.
    # encoded_patches = PatchEncoder(num_patches, projection_dim)(x)
    #
    # # Create multiple layers of the Transformer block.
    # for i in range(transformer_layers):
    #     # Layer normalization 1.
    #     x1 = LayerNormalization(epsilon=1e-6, name="ln_" + str(i) + "_1")(encoded_patches)
    #     # Create a multi-head attention layer.
    #     attention_output = MultiHeadAttention(
    #         num_heads=num_heads, key_dim=projection_dim, dropout=0.1, name="mha_" + str(i)
    #     )(x1, x1)
    #     # Skip connection 1.
    #     x2 = Add()([attention_output, encoded_patches])
    #     # Layer normalization 2.
    #     x3 = LayerNormalization(epsilon=1e-6, name="ln_" + str(i) + "_2")(x2)
    #     # MLP.
    #     x3 = mlp(x3, hidden_units=transformer_units, dropout_rate=0.1, name="mlp_" + str(i))
    #     # Skip connection 2.
    #     encoded_patches = Add()([x3, x2])
    #
    # # Create a [batch_size, projection_dim] tensor.
    # representation = LayerNormalization(epsilon=1e-6, name="ln_rep")(encoded_patches)
    # representation = Flatten()(representation)
    # representation = Dropout(0.2)(representation)

    representation = Flatten()(x)
    # Compress
    compress_dim = 2000
    x = Dense(compress_dim, name="latent_vector")(representation)
    x = LeakyReLU(alpha=0.1)(x)
    x = Dropout(0.5, input_shape=(None, compress_dim))(x)


    outs = []
    for i in range(cell_num):
        ol = Dense(num_regions, use_bias=False, name="out_row_"+str(i))(x)
        ol = LeakyReLU(alpha=0.1, name="act_out_row_"+str(i))(ol)
        outs.append(ol)
        if i % 50 == 0:
            print(i, end=" ")

    # x = Dense(cell_num * num_regions)(representation)
    # x = LeakyReLU(alpha=0.1)(x)
    # outputs = Reshape((cell_num, num_regions))(x)
    outputs = tf.stack(outs, axis=1)
    print(outputs)
    model = Model(inputs, outputs, name="model")
    print("\nModel constructed")
    return model


def resnet_layer(inputs,
                 num_filters=16,
                 kernel_size=3,
                 strides=1,
                 activation='relu',
                 batch_normalization=True,
                 conv_first=True,
                 name="rl_"):

    conv = Conv1D(num_filters,
                  kernel_size=kernel_size,
                  strides=strides,
                  padding='same',
                  # use_bias=False,
                  name=name+"conv1d"
                  # kernel_regularizer=l2(1e-6),
                  # activity_regularizer=l2(1e-6)
                  )

    x = inputs
    if conv_first:
        x = conv(x)
        if batch_normalization:
            x = BatchNormalization(name=name+"bn")(x)
        if activation is not None:
            x = LeakyReLU(alpha=0.1, name=name+"act")(x)
    else:
        if batch_normalization:
            x = BatchNormalization(name=name+"bn")(x)
        if activation is not None:
            x = LeakyReLU(alpha=0.1, name=name+"act")(x)
        x = conv(x)
    return x


def resnet_v2(input_x, num_stages, num_res_blocks):
    # Start model definition.
    num_filters_in = 128

    # v2 performs Conv2D with BN-ReLU on input before splitting into 2 paths
    x = resnet_layer(inputs=input_x,
                     num_filters=num_filters_in,
                     conv_first=True, name="rl_1_")

    # Instantiate the stack of residual units
    for stage in range(num_stages):
        for res_block in range(num_res_blocks):
            cname = "rl_" + str(stage) + "_" + str(res_block) + "_"
            activation = 'relu'
            batch_normalization = True
            strides = 1
            num_filters_out = int(num_filters_in * 1.1)
            if stage == 0:
                if res_block == 0:  # first layer and first stage
                    activation = None
                    batch_normalization = False
            else:
                if res_block == 0:  # first layer but not first stage
                    strides = 2    # downsample

            # bottleneck residual unit
            y = resnet_layer(inputs=x,
                             num_filters=num_filters_in,
                             kernel_size=1,
                             strides=strides,
                             activation=activation,
                             batch_normalization=batch_normalization,
                             conv_first=False,
                             name=cname+"1_")
            y = resnet_layer(inputs=y,
                             num_filters=num_filters_in,
                             conv_first=False,
                             name=cname+"2_")
            y = resnet_layer(inputs=y,
                             num_filters=num_filters_out,
                             kernel_size=1,
                             conv_first=False,
                             name=cname+"3_")
            if res_block == 0:
                # linear projection residual shortcut connection to match
                # changed dims
                x = resnet_layer(inputs=x,
                                 num_filters=num_filters_out,
                                 kernel_size=1,
                                 strides=strides,
                                 activation=None,
                                 batch_normalization=False,
                                 name=cname+"4_")
            x = Add()([x, y])

        num_filters_in = num_filters_out

    x = BatchNormalization(name="res_bn_final")(x)
    x = LeakyReLU(alpha=0.1, name="res_act_final")(x)

    return x


def mlp(x, hidden_units, dropout_rate, name):
    for units in hidden_units:
        x = Dense(units, name=name + str(units))(x)
        x = LeakyReLU(alpha=0.1)(x)
        x = Dropout(dropout_rate)(x)
    return x


class PatchEncoder(Layer):
    def __init__(self, num_patches, projection_dim, **kwargs):
        super(PatchEncoder, self).__init__(**kwargs)
        self.num_patches = num_patches
        self.projection = Dense(units=projection_dim, name="projection_dense")
        self.projection_dim = projection_dim
        self.position_embedding = Embedding(
            input_dim=num_patches, output_dim=projection_dim, name="pos_embedding"
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

def keras_model_memory_usage_in_bytes(model, *, batch_size: int):
    """
    Return the estimated memory usage of a given Keras model in bytes.
    This includes the model weights and layers, but excludes the dataset.

    The model shapes are multipled by the batch size, but the weights are not.

    Args:
        model: A Keras model.
        batch_size: The batch size you intend to run the model with. If you
            have already specified the batch size in the model itself, then
            pass `1` as the argument here.
    Returns:
        An estimate of the Keras model's memory usage in bytes.

    """
    default_dtype = tf.keras.backend.floatx()
    shapes_mem_count = 0
    internal_model_mem_count = 0
    for layer in model.layers:
        if isinstance(layer, tf.keras.Model):
            internal_model_mem_count += keras_model_memory_usage_in_bytes(
                layer, batch_size=batch_size
            )
        single_layer_mem = tf.as_dtype(layer.dtype or default_dtype).size
        out_shape = layer.output_shape
        if isinstance(out_shape, list):
            out_shape = out_shape[0]
        for s in out_shape:
            if s is None:
                continue
            single_layer_mem *= s
        shapes_mem_count += single_layer_mem

    trainable_count = sum(
        [tf.keras.backend.count_params(p) for p in model.trainable_weights]
    )
    non_trainable_count = sum(
        [tf.keras.backend.count_params(p) for p in model.non_trainable_weights]
    )

    total_memory = (
        batch_size * shapes_mem_count
        + internal_model_mem_count
        + trainable_count
        + non_trainable_count
    )
    return cm.get_human_readable(total_memory)
