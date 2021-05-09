import os

# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = '0'
from tensorflow.keras import regularizers
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Reshape
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras import layers
import tensorflow as tf
import numpy as np
import time
import attention
import transformer_layers as tl
from scipy import stats


def simple_model(input_size, num_regions, cell_num):
    latent_dim = 128
    l1_weight = 1e-8
    input_shape = (num_regions, input_size, 4)
    inputs = Input(shape=input_shape)
    x = inputs
    xd = Dropout(0.5, input_shape=(None, num_regions, input_size, 4))(x)
    x = xd
    x = layers.Conv1D(128, 3, padding='same', activation='relu')(x)
    x = layers.Conv1D(128, 3, padding='same', activation='relu')(x)
    # x = Dropout(0.5, input_shape=(None, num_regions, input_size, 128))(x)
    x = Flatten()(x)
    # x = Dense(latent_dim, use_bias=False, activity_regularizer=regularizers.l1(l1_weight))(x)
    #
    # x = Dense(128)(x)
    # x = LeakyReLU(alpha=0.2)(x)
    # x = Dense(128)(x)
    # x = LeakyReLU(alpha=0.2)(x)

    x = Dense(cell_num * num_regions)(x)
    outputs = Reshape((num_regions, cell_num))(x)
    model = Model(inputs, outputs, name="model")
    return model


BATCH_SIZE = 137
units = 512
num_layers = 4
d_model = 1184
dff = 512
num_heads = 8
dropout_rate = 0.1
EPOCHS = 2

def attention_model(dataset, num_cells, num_regions, prev_model=None):
    dataset = dataset.batch(BATCH_SIZE, drop_remainder=True)
    steps_per_epoch = len(dataset)
    if prev_model == None:
        learning_rate = tl.CustomSchedule(d_model)
        optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98,
                                             epsilon=1e-9)
        transformer = tl.Transformer(
            num_layers=num_layers,
            d_model=d_model,
            num_heads=num_heads,
            dff=dff,
            target_vocab_size=num_cells,
            pe_input=1000,
            pe_target=1000,
            rate=dropout_rate)
    else:
        transformer = prev_model[0]
        optimizer = prev_model[1]

    checkpoint_path = "./training_checkpoints"

    ckpt = tf.train.Checkpoint(transformer=transformer,
                               optimizer=optimizer)

    ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)

    # if a checkpoint exists, restore the latest checkpoint.
    if ckpt_manager.latest_checkpoint:
        ckpt.restore(ckpt_manager.latest_checkpoint)
        print('Latest checkpoint restored!!')
        # return transformer, optimizer

    def loss_function(real, pred):
        # mask = tf.math.logical_not(tf.math.equal(real, 0))
        loss_ = tf.keras.losses.mean_squared_error(real, pred)

        # mask = tf.cast(mask, dtype=loss_.dtype)
        # loss_ *= mask

        return tf.reduce_mean(loss_)


    @tf.function()
    def train_step(inp, tar):
        tar_inp = tar[:, :-1]
        tar_real = tar[:, 1:]

        with tf.GradientTape() as tape:
            predictions, _ = transformer(inp, tar_inp,
                                         True)
            loss = loss_function(tar_real, predictions)

        gradients = tape.gradient(loss, transformer.trainable_variables)
        optimizer.apply_gradients(zip(gradients, transformer.trainable_variables))
        return loss

    for epoch in range(EPOCHS):
        total_loss = 0

        # inp -> portuguese, tar -> english
        for (batch, (inp, tar)) in enumerate(dataset.take(steps_per_epoch)):
            batch_loss = train_step(inp, tar)
            total_loss += batch_loss
            if batch % 100 == 0:
                print(f'Epoch {epoch + 1} Batch {batch} Loss {batch_loss.numpy():.4f}')

        print(f'Epoch {epoch + 1} Loss {total_loss / steps_per_epoch:.4f}')
        ckpt_save_path = ckpt_manager.save()

    return transformer, optimizer


def evaluate(test_dataset, transformer, num_regions, num_cells, cheating):
    results = []
    test_dataset = test_dataset.batch(BATCH_SIZE, drop_remainder=True)
    for batch in iter(test_dataset):
        print("-", end="")
        bs = len(batch)
        # output = tf.zeros((bs, 1, num_cells))
        output = cheating[:, :1, :]
        for t in range(1, num_regions, 1):
            # predictions.shape == (batch_size, seq_len, vocab_size)
            predictions, attention_weights = transformer(batch,
                                                         output,
                                                         False)

            # select the last word from the seq_len dimension
            predictions = predictions[:, -1:, :]  # (batch_size, 1, vocab_size)

            # concatentate the predicted_id to the output which is given to the decoder
            # as its input.
            output = tf.concat([output, predictions], axis=1)
        corr = stats.pearsonr(output.numpy()[:, :1, :].flatten(), cheating.numpy()[:, :1, :].flatten())[0]
        print(corr)
        result = output.numpy().squeeze()
        results.extend(result)
    print()
    return results
