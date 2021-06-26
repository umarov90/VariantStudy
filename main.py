import os

import joblib

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = '2'
import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)
import tensorflow as tf
import gc
import random
import pandas as pd
import math
import attribution
import model as mo
import numpy as np
import common as cm
from pathlib import Path
import pickle
from tensorflow.keras.optimizers import Adam
from scipy import stats
import matplotlib
from tensorflow.keras import backend as K
import matplotlib.pyplot as plt
from heapq import nsmallest
import copy
import seaborn as sns
import shutil
import parse_data as parser
from datetime import datetime
matplotlib.use("agg")
from scipy.ndimage.filters import gaussian_filter
from tensorflow.keras import mixed_precision
mixed_precision.set_global_policy('mixed_float16')

def train():
    # Apply smoothing to output bins? Ask Hon how to do it best
    # transformer_layers = 8 Try 1 instead of 8
    model_folder = "model1"
    model_name = "expression_model_1.h5"
    figures_folder = "figures_1"
    input_size = 500000
    half_size = input_size / 2
    max_shift = 2000
    bin_size = 1000
    hic_bin_size = 20000
    num_hic_bins = int(input_size / hic_bin_size)
    num_regions = int(input_size / bin_size)
    mid_bin = math.floor(num_regions / 2)
    BATCH_SIZE = 1
    # strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy()
    strategy = tf.distribute.MirroredStrategy()
    print('Number of devices: {}'.format(strategy.num_replicas_in_sync))
    GLOBAL_BATCH_SIZE = BATCH_SIZE * strategy.num_replicas_in_sync
    STEPS_PER_EPOCH = 4000
    out_stack_num = 1000
    num_epochs = 10000
    test_chr = "chr1"

    Path(model_folder).mkdir(parents=True, exist_ok=True)
    Path(figures_folder + "/" + "attribution").mkdir(parents=True, exist_ok=True)
    Path(figures_folder + "/" + "tracks").mkdir(parents=True, exist_ok=True)

    chromosomes = ["chrX", "chrY"]
    # our_model = mo.simple_model(input_size, num_regions, out_stack_num)
    # aaa = mo.keras_model_memory_usage_in_bytes(our_model, batch_size=1)
    # print(aaa)
    for i in range(1, 23):
        chromosomes.append("chr" + str(i))

    hic_keys = parser.parse_hic()
    ga, one_hot, train_info, test_info, test_seq = parser.get_sequences(input_size, bin_size, chromosomes)
    gas_keys = parser.parse_tracks(ga, bin_size)

    print("Number of tracks: " + str(len(gas_keys)))
    with strategy.scope():
        if Path(model_folder + "/" + model_name).is_file():
            our_model = tf.keras.models.load_model(model_folder + "/" + model_name,
                                                   custom_objects={'PatchEncoder': mo.PatchEncoder})
            print('Loaded existing model')
        else:
            our_model = mo.simple_model(input_size, num_regions, out_stack_num)
            Path(model_folder).mkdir(parents=True, exist_ok=True)
            our_model.save(model_folder + "/" + model_name)
            print("Model saved")
            joblib.dump(our_model.get_layer("out_row_0").get_weights(), model_folder + "/" + gas_keys[0], compress=3)
            for i in range(1, len(gas_keys), 1):
                shutil.copyfile(model_folder + "/" + gas_keys[0], model_folder + "/" + gas_keys[i])
            print("\nWeights saved")
    # print("0000000000000000000000000000")
    # our_model_new = mo.simple_model(input_size, num_regions, 200)
    # for l in our_model_new.layers:
    #     if "out_row" not in l.name:
    #         try:
    #             l.set_weights(our_model.get_layer(l.name).get_weights())
    #         except Exception as e:
    #             print(l.name)
    # our_model = our_model_new
    # print("0000000000000000000000000000")
    for k in range(num_epochs):
        print(datetime.now().strftime('[%H:%M:%S] ') + "Epoch " + str(k))
        if k > 0:
            with strategy.scope():
                our_model = tf.keras.models.load_model(model_folder + "/" + model_name,
                                                       custom_objects={'PatchEncoder': mo.PatchEncoder})
        input_sequences = []
        output_scores = []
        print(datetime.now().strftime('[%H:%M:%S] ') + "Preparing sequences")
        chosen_tracks = random.sample(gas_keys, out_stack_num - len(hic_keys))
        gas = {}
        for i, key in enumerate(chosen_tracks):
            our_model.get_layer("out_row_" + str(i)).set_weights(joblib.load(model_folder + "/" + key))
            gas[key] = joblib.load("parsed_tracks/" + key)
        print(datetime.now().strftime('[%H:%M:%S] ') + "Loaded the tracks")
        err = 0
        rands = []
        for i, info in enumerate(train_info):
            if len(output_scores) >= GLOBAL_BATCH_SIZE * STEPS_PER_EPOCH:
                break
            if i % 500 == 0:
                print(i, end=" ")
                gc.collect()
            try:
                rand_var = random.randint(0, max_shift)
                start = int(info[1] + (rand_var - max_shift / 2) - half_size)
                ns = one_hot[info[0]][start:start + input_size]
                if info[0] not in chromosomes or len(ns) == input_size:
                    rands.append(rand_var)
                else:
                    rands.append(-1)
                    continue
                start_bin = int(start / bin_size)
                scores = []
                for key in chosen_tracks:
                    scores.append(gas[key][info[0]][start_bin: start_bin + num_regions])
                input_sequences.append(ns)
                output_scores.append(scores)
            except Exception as e:
                print(e)
                err += 1
        print(np.asarray(input_sequences).shape)
        print(np.asarray(output_scores).shape)
        print("\nHi-C")
        for key in hic_keys:
            print(key, end=" ")
            hdf = joblib.load("parsed_hic/" + key)
            ni = 0
            for i, info in enumerate(train_info):
                if i >= len(rands):
                    break
                try:
                    rand_var = rands[i]
                    if rand_var == -1:
                        continue
                    hd = hdf[info[0]]
                    hic_mat = np.zeros((num_hic_bins, num_hic_bins))
                    start_hic = int((info[1] + (rand_var - max_shift / 2) - half_size))
                    end_hic = start_hic + input_size
                    start_row = hd['locus1'].searchsorted(start_hic - hic_bin_size, side='left')
                    end_row = hd['locus1'].searchsorted(end_hic, side='right')
                    hd = hd.iloc[start_row:end_row]
                    # convert start of the input region to the bin number
                    start_hic = int(start_hic / hic_bin_size)
                    # subtract start bin from the binned entries in the range [start_row : end_row]
                    l1 = (np.floor(hd["locus1"].values / hic_bin_size) - start_hic).astype(int)
                    l2 = (np.floor(hd["locus2"].values / hic_bin_size) - start_hic).astype(int)
                    hic_score = hd["score"].values
                    # drop contacts with regions outside the [start_row : end_row] range
                    lix = (l2 < len(hic_mat)) & (l2 >= 0) & (l1 >= 0)
                    l1 = l1[lix]
                    l2 = l2[lix]
                    hic_score = hic_score[lix]
                    hic_mat[l1, l2] += hic_score
                    # hic_mat = hic_mat + hic_mat.T - np.diag(np.diag(hic_mat))
                    hic_mat = gaussian_filter(hic_mat, sigma=1)
                    # print(f"original {len(hic_mat.flatten())}")
                    hic_mat = hic_mat[np.triu_indices_from(hic_mat, k=1)]
                    # print(f"triu {len(hic_mat.flatten())}")
                    hic_mat.resize(num_regions, refcheck=False)
                    # print(f"resize {len(hic_mat.flatten())}")
                    output_scores[ni].append(hic_mat)
                    ni += 1
                except Exception as e:
                    print(e)
                    err += 1
        print(np.asarray(input_sequences).shape)
        print(np.asarray(output_scores).shape)
        print("\nTest output")
        # preparing test output tracks
        test_output = []
        for i in range(len(test_info)):
            scores = []
            start = int((test_info[i][1] - half_size) / bin_size)
            for key in chosen_tracks:
                scores.append(gas[key][test_info[i][0]][start: start + num_regions])
            test_output.append(scores)
        for key in hic_keys:
            print(key, end=" ")
            hdf = joblib.load("parsed_hic/" + key)
            for i, info in enumerate(test_info):
                try:
                    hd = hdf[info[0]]
                    hic_mat = np.zeros((num_hic_bins, num_hic_bins))
                    start_hic = int((info[1] - half_size))
                    end_hic = start_hic + input_size
                    start_row = hd['locus1'].searchsorted(start_hic - hic_bin_size, side='left')
                    end_row = hd['locus1'].searchsorted(end_hic, side='right')
                    hd = hd.iloc[start_row:end_row]
                    # convert start of the input region to the bin number
                    start_hic = int(start_hic / hic_bin_size)
                    # subtract start bin from the binned entries in the range [start_row : end_row]
                    l1 = (np.floor(hd["locus1"].values / hic_bin_size) - start_hic).astype(int)
                    l2 = (np.floor(hd["locus2"].values / hic_bin_size) - start_hic).astype(int)
                    hic_score = hd["score"].values
                    # drop contacts with regions outside the [start_row : end_row] range
                    lix = (l2 < len(hic_mat)) & (l2 >= 0) & (l1 >= 0)
                    l1 = l1[lix]
                    l2 = l2[lix]
                    hic_score = hic_score[lix]
                    hic_mat[l1, l2] += hic_score
                    # hic_mat = hic_mat + hic_mat.T - np.diag(np.diag(hic_mat))
                    hic_mat = gaussian_filter(hic_mat, sigma=1)
                    # print(f"original {len(hic_mat.flatten())}")
                    hic_mat = hic_mat[np.triu_indices_from(hic_mat, k=1)]
                    # print(f"triu {len(hic_mat.flatten())}")
                    hic_mat.resize(num_regions, refcheck=False)
                    # print(f"resize {len(hic_mat.flatten())}")
                    test_output[i].append(hic_mat)
                except Exception as e:
                    print(e)
                    err += 1

        test_output = np.asarray(test_output)
        print(test_output.shape)

        print("")
        print(datetime.now().strftime('[%H:%M:%S] ') + "Problems: " + str(err))
        output_scores = np.asarray(output_scores)
        input_sequences = np.asarray(input_sequences)

        rng_state = np.random.get_state()
        np.random.shuffle(input_sequences)
        np.random.set_state(rng_state)
        np.random.shuffle(output_scores)

        # input_sequences = input_sequences[:GLOBAL_BATCH_SIZE * STEPS_PER_EPOCH]
        # output_scores = output_scores[:GLOBAL_BATCH_SIZE * STEPS_PER_EPOCH]

        print(datetime.now().strftime('[%H:%M:%S] ') + "Compiling model")
        # if k < 300:
        #     lr = 0.0001
        # elif k < 600:
        #     lr = 0.00005
        # else:
        #     lr = 0.00002
        lr = 0.0001
        fit_epochs = 1
        with strategy.scope():
            if k == 0:
                fit_epochs = 10
            # if k % 9 != 0:
            #     freeze = True
            #     fit_epochs = 4
            # else:
            #     freeze = False
            #     fit_epochs = 2
            for l in our_model.layers:
                # if "out_row" not in l.name and freeze:
                #     l.trainable = False
                # else:
                l.trainable = True
            our_model.compile(loss="mse", optimizer=Adam(learning_rate=lr))

        print(datetime.now().strftime('[%H:%M:%S] ') + "Training")
        # if k != 0:
        try:
            train_data = wrap(input_sequences, output_scores, GLOBAL_BATCH_SIZE)
            our_model.fit(train_data, epochs=fit_epochs)
            our_model.save(model_folder + "/" + model_name + "_temp.h5")
            os.remove(model_folder + "/" + model_name)
            os.rename(model_folder + "/" + model_name + "_temp.h5", model_folder + "/" + model_name)
            for i, key in enumerate(chosen_tracks):
                joblib.dump(our_model.get_layer("out_row_" + str(i)).get_weights(), model_folder + "/" + key + "_temp",
                            compress=3)
                os.remove(model_folder + "/" + key)
                os.rename(model_folder + "/" + key + "_temp", model_folder + "/" + key)
        except Exception as e:
            print(e)
            print(datetime.now().strftime('[%H:%M:%S] ') + "Error while training. Loading previous model.")
            with strategy.scope():
                our_model = tf.keras.models.load_model(model_folder + "/" + model_name,
                                                       custom_objects={'PatchEncoder': mo.PatchEncoder})
            del input_sequences
            del output_scores
            del predictions
            gc.collect()
            continue

        if k % 10 == 0 : # and k != 0
            print(datetime.now().strftime('[%H:%M:%S] ') + "Evaluating")
            try:
                print("Training set")
                predictions = our_model.predict(input_sequences[0:1000], batch_size=1)

                corrs = []
                for it, ct in enumerate(chosen_tracks):
                    if "ctss" in ct:
                        a = []
                        b = []
                        for i in range(len(predictions)):
                            a.append(predictions[i][it][mid_bin])
                            b.append(output_scores[i][it][mid_bin])
                        corrs.append(stats.spearmanr(a, b)[0])
                print(f"Correlation : {np.mean(corrs)}")

                print("Drawing tracks")
                pic_count = 0
                for it, ct in enumerate(chosen_tracks):
                    for i in range(len(predictions)):
                        if np.sum(output_scores[i][it]) == 0:
                            continue
                        fig, axs = plt.subplots(2, 1, figsize=(12, 8))
                        vector1 = predictions[i][it]
                        vector2 = output_scores[i][it]
                        x = range(num_regions)
                        d1 = {'bin': x, 'expression': vector1}
                        df1 = pd.DataFrame(d1)
                        d2 = {'bin': x, 'expression': vector2}
                        df2 = pd.DataFrame(d2)
                        sns.lineplot(data=df1, x='bin', y='expression', ax=axs[0])
                        axs[0].set_title("Prediction")
                        sns.lineplot(data=df2, x='bin', y='expression', ax=axs[1])
                        axs[1].set_title("Ground truth")
                        fig.tight_layout()
                        plt.savefig(figures_folder + "/tracks/train_track_" + str(i + 1) + "_" + str(ct) + ".png")
                        plt.close(fig)
                        pic_count += 1
                        break
                    if pic_count > 10:
                        break
                print("Drawing contact maps")
                for h in range(len(hic_keys)):
                    pic_count = 0
                    it = len(chosen_tracks) + h
                    for i in range(len(predictions)):
                        if np.sum(output_scores[i][it]) == 0:
                            continue
                        mat_gt = recover_shape(output_scores[i][it], num_hic_bins)
                        mat_pred = recover_shape(predictions[i][it], num_hic_bins)
                        fig, axs = plt.subplots(2, 1, figsize=(8, 8))
                        sns.heatmap(mat_pred, linewidth=0.0, ax=axs[0])
                        axs[0].set_title("Prediction")
                        sns.heatmap(mat_gt, linewidth=0.0, ax=axs[1])
                        axs[1].set_title("Ground truth")
                        plt.tight_layout()
                        plt.savefig(figures_folder + "/hic/train_track_" + str(i + 1) + "_" + str(hic_keys[h]) + ".png")
                        plt.close(fig)
                        pic_count += 1
                        if pic_count > 4:
                            break

                print("Test set")
                predictions = our_model.predict(test_seq[0:1000], batch_size=1)

                corrs = []
                for it, ct in enumerate(chosen_tracks):
                    if "ctss" in ct:
                        a = []
                        b = []
                        for i in range(len(predictions)):
                            a.append(predictions[i][it][mid_bin])
                            b.append(test_output[i][it][mid_bin])
                        corrs.append(stats.spearmanr(a, b)[0])
                print(f"Correlation : {np.mean(corrs)}")

                with open("result.txt", "a+") as myfile:
                    myfile.write(str(np.mean(corrs)))

                print("Drawing tracks")
                pic_count = 0
                for it, ct in enumerate(chosen_tracks):
                    for i in range(len(predictions)):
                        if np.sum(output_scores[i][it]) == 0:
                            continue
                        fig, axs = plt.subplots(2, 1, figsize=(12, 8))
                        vector1 = predictions[i][it]
                        vector2 = test_output[i][it]
                        x = range(num_regions)
                        d1 = {'bin': x, 'expression': vector1}
                        df1 = pd.DataFrame(d1)
                        d2 = {'bin': x, 'expression': vector2}
                        df2 = pd.DataFrame(d2)
                        sns.lineplot(data=df1, x='bin', y='expression', ax=axs[0])
                        axs[0].set_title("Prediction")
                        sns.lineplot(data=df2, x='bin', y='expression', ax=axs[1])
                        axs[1].set_title("Ground truth")
                        fig.tight_layout()
                        plt.savefig(figures_folder + "/tracks/test_track_" + str(i + 1) + "_" + str(ct) + ".png")
                        plt.close(fig)
                        pic_count += 1
                        break
                    if pic_count > 10:
                        break

                print("Drawing contact maps")
                for h in range(len(hic_keys)):
                    pic_count = 0
                    it = len(chosen_tracks) + h
                    for i in range(len(predictions)):
                        if np.sum(output_scores[i][it]) == 0:
                            continue
                        mat_gt = recover_shape(output_scores[i][it], num_hic_bins)
                        mat_pred = recover_shape(predictions[i][it], num_hic_bins)
                        fig, axs = plt.subplots(2, 1, figsize=(8, 8))
                        sns.heatmap(mat_pred, linewidth=0.0, ax=axs[0])
                        axs[0].set_title("Prediction")
                        sns.heatmap(mat_gt, linewidth=0.0, ax=axs[1])
                        axs[1].set_title("Ground truth")
                        plt.tight_layout()
                        plt.savefig(figures_folder + "/hic/test_track_" + str(i + 1) + "_" + str(hic_keys[h]) + ".png")
                        plt.close(fig)
                        pic_count += 1
                        if pic_count > 4:
                            break

                # Gene regplot
                # for c, cell in enumerate(cells):
                #     ci = -2 + c
                #     a = []
                #     b = []
                #     for i in range(len(predictions)):
                #         if test_class[i] == 0:
                #             continue
                #         a.append(predictions[i][ci][mid_bin])
                #         b.append(test_output[i][ci][mid_bin])
                #
                #     pickle.dump(a, open(figures_folder + "/" + str(cell) + "_a" + str(k) + ".p", "wb"),
                #                 protocol=pickle.HIGHEST_PROTOCOL)
                #     pickle.dump(b, open(figures_folder + "/" + str(cell) + "_b" + str(k) + ".p", "wb"),
                #                 protocol=pickle.HIGHEST_PROTOCOL)
                #
                #     fig, ax = plt.subplots(figsize=(6, 6))
                #     r, p = stats.spearmanr(a, b)
                #
                #     sns.regplot(x=a, y=b,
                #                 ci=None, label="r = {0:.2f}; p = {1:.2e}".format(r, p)).legend(loc="best")
                #
                #     ax.set(xlabel='Predicted', ylabel='Ground truth')
                #     plt.title("Gene expression prediction")
                #     fig.tight_layout()
                #     plt.savefig(figures_folder + "/corr_" + str(k) + "_" + str(cell) + ".svg")
                #     plt.close(fig)

                # attribution
                # for c, cell in enumerate(cells):
                #     for i in range(1200, 1210, 1):
                #         baseline = tf.zeros(shape=(input_size, 4))
                #         image = test_input_sequences[i].astype('float32')
                #         ig_attributions = attribution.integrated_gradients(our_model, baseline=baseline,
                #                                                            image=image,
                #                                                            target_class_idx=[mid_bin, c],
                #                                                            m_steps=40)
                #
                #         attribution_mask = tf.squeeze(ig_attributions).numpy()
                #         attribution_mask = (attribution_mask - np.min(attribution_mask)) / (
                #                     np.max(attribution_mask) - np.min(attribution_mask))
                #         attribution_mask = np.mean(attribution_mask, axis=-1, keepdims=True)
                #         attribution_mask[int(input_size / 2) - 2000 : int(input_size / 2) + 2000, :] = np.nan
                #         attribution_mask = skimage.measure.block_reduce(attribution_mask, (100, 1), np.mean)
                #         attribution_mask = np.transpose(attribution_mask)
                #
                #         fig, ax = plt.subplots(figsize=(60, 6))
                #         sns.heatmap(attribution_mask, linewidth=0.0, ax=ax)
                #         plt.tight_layout()
                #         plt.savefig(figures_folder + "/attribution/track_" + str(i + 1) + "_" + str(cell) + "_" + test_info[i] + ".jpg")
                #         plt.close(fig)
                del predictions
            except Exception as e:
                print(e)
                print(datetime.now().strftime('[%H:%M:%S] ') + "Problem during evaluation")
        print(datetime.now().strftime('[%H:%M:%S] ') + "Cleaning")
        # Needed to prevent Keras memory leak
        del input_sequences
        del output_scores
        del our_model
        del gas
        gc.collect()
        K.clear_session()
        tf.compat.v1.reset_default_graph()
        print(datetime.now().strftime('[%H:%M:%S] ') + "Epoch " + str(k) + " finished. ")


def recover_shape(v, size_X):
    end = int( (size_X * size_X - size_X) / 2)
    v = v[:end]
    X = np.zeros((size_X, size_X))
    X[np.triu_indices(X.shape[0], k=1)] = v
    X = X + X.T
    return X


def wrap(input_sequences, output_scores, bs):
    train_data = tf.data.Dataset.from_tensor_slices((input_sequences, output_scores))
    train_data = train_data.batch(bs)
    options = tf.data.Options()
    options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA
    train_data = train_data.with_options(options)
    return train_data


if __name__ == '__main__':
    os.chdir(open("data_dir").read().strip())
    train()
