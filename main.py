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
import time
import attribution
import model as mo
import numpy as np
import common as cm
from pathlib import Path
import pickle
from scipy import stats
import matplotlib
from tensorflow.keras import backend as K
import tensorflow_addons as tfa
import matplotlib.pyplot as plt
from heapq import nsmallest
import copy
import seaborn as sns
import shutil
import psutil
import sys
import parse_data as parser
from datetime import datetime
matplotlib.use("agg")
from scipy.ndimage.filters import gaussian_filter
from tensorflow.keras import mixed_precision
mixed_precision.set_global_policy('mixed_float16')

seed = 0
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)


def train():
    # Apply smoothing to output bins? Ask Hon how to do it best
    # Turn off bias for all conv1d and try perf after 20.
    model_folder = "model1"
    model_name = "expression_model_1.h5"
    figures_folder = "figures_1"
    input_size = 40000
    half_size = input_size / 2
    max_shift = 200
    bin_size = 200
    hic_bin_size = 10000
    num_hic_bins = int(input_size / hic_bin_size)
    num_regions = int(input_size / bin_size)
    mid_bin = math.floor(num_regions / 2)
    BATCH_SIZE = 1
    strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy()
    # strategy = tf.distribute.MirroredStrategy()
    print('Number of devices: {}'.format(strategy.num_replicas_in_sync))
    GLOBAL_BATCH_SIZE = BATCH_SIZE * strategy.num_replicas_in_sync
    STEPS_PER_EPOCH = 4000
    out_stack_num = 1000
    num_epochs = 10000
    test_chr = "chr1"
    hic_track_size = 1

    Path(model_folder).mkdir(parents=True, exist_ok=True)
    Path(figures_folder + "/" + "attribution").mkdir(parents=True, exist_ok=True)
    Path(figures_folder + "/" + "tracks").mkdir(parents=True, exist_ok=True)
    Path(figures_folder + "/" + "hic").mkdir(parents=True, exist_ok=True)

    chromosomes = ["chrX"] # "chrY"
    # our_model = mo.simple_model(input_size, num_regions, out_stack_num)
    # aaa = mo.keras_model_memory_usage_in_bytes(our_model, batch_size=1)
    # print(aaa)
    for i in range(1, 23):
        chromosomes.append("chr" + str(i))

    # hic_keys = parser.parse_hic()
    # hic_keys = ["hic_ADAC418_10kb_interactions.txt.bz2"]
    ga, one_hot, train_info, test_info, test_seq = parser.get_sequences(input_size, bin_size, chromosomes)
    if Path("pickle/gas_keys.gz").is_file():
        gas_keys = joblib.load("pickle/gas_keys.gz")
    else:
        gas_keys = parser.parse_tracks(ga, bin_size)

    print("Number of tracks: " + str(len(gas_keys)))
    model_was_created = True
    with strategy.scope():
        if Path(model_folder + "/" + model_name).is_file():
            our_model = tf.keras.models.load_model(model_folder + "/" + model_name,
                                                   custom_objects={'PatchEncoder': mo.PatchEncoder})
            print('Loaded existing model')
            model_was_created = False
        else:
            our_model = mo.simple_model(input_size, num_regions, out_stack_num)
            Path(model_folder).mkdir(parents=True, exist_ok=True)
            our_model.save(model_folder + "/" + model_name)
            print("Model saved")
            # joblib.dump(our_model.get_layer("out_row_0").get_weights(), model_folder + "/" + gas_keys[0], compress=3)
            # for i in range(1, len(gas_keys), 1):
            #     shutil.copyfile(model_folder + "/" + gas_keys[0], model_folder + "/" + gas_keys[i])
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
    print_memory()
    for name, size in sorted(((name, sys.getsizeof(value)) for name, value in locals().items()),
                             key=lambda x: -x[1])[:10]:
        print("{:>30}: {:>8}".format(name, cm.get_human_readable(size)))

    # gast = parser.parse_one_track(ga, bin_size, "tracks/CAGE.RNA.ctss.adrenal_gland_adult_pool1.CNhs11793.FANTOM5.100nt.bed.gz")
    # for ti in test_info:
    #     starttt = int(ti[1] / bin_size)
    #     aaa = gast[ti[0]][starttt]
    #     print(aaa)

    for k in range(num_epochs):
        print(datetime.now().strftime('[%H:%M:%S] ') + "Epoch " + str(k))
        if k > 0:
            with strategy.scope():
                our_model = tf.keras.models.load_model(model_folder + "/" + model_name,
                                                       custom_objects={'PatchEncoder': mo.PatchEncoder})

        # rng_state = np.random.get_state()
        # np.random.shuffle(input_sequences)
        # np.random.set_state(rng_state)
        # np.random.shuffle(output_scores)
        random.shuffle(train_info)

        input_sequences = []
        output_scores = []
        print(datetime.now().strftime('[%H:%M:%S] ') + "Preparing sequences")
        chosen_tracks = random.sample(gas_keys, out_stack_num) # - len(hic_keys) * (hic_track_size)
        gas = {}
        for i, key in enumerate(chosen_tracks):
            if k > 0:
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
                if info[0] not in chromosomes:
                    rands.append(-1)
                    continue
                ns = one_hot[info[0]][start:start + input_size]
                if len(ns) == input_size:
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
        # print(np.asarray(input_sequences).shape)
        # print(np.asarray(output_scores).shape)
        # print("\nHi-C")
        # for key in hic_keys:
        #     print(key, end=" ")
        #     hdf = joblib.load("parsed_hic/" + key)
        #     ni = 0
        #     for i, info in enumerate(train_info):
        #         if i >= len(rands):
        #             break
        #         try:
        #             rand_var = rands[i]
        #             if rand_var == -1:
        #                 continue
        #             hd = hdf[info[0]]
        #             hic_mat = np.zeros((num_hic_bins, num_hic_bins))
        #             start_hic = int((info[1] + (rand_var - max_shift / 2) - half_size))
        #             end_hic = start_hic + input_size
        #             start_row = hd['locus1'].searchsorted(start_hic - hic_bin_size, side='left')
        #             end_row = hd['locus1'].searchsorted(end_hic, side='right')
        #             hd = hd.iloc[start_row:end_row]
        #             # convert start of the input region to the bin number
        #             start_hic = int(start_hic / hic_bin_size)
        #             # subtract start bin from the binned entries in the range [start_row : end_row]
        #             l1 = (np.floor(hd["locus1"].values / hic_bin_size) - start_hic).astype(int)
        #             l2 = (np.floor(hd["locus2"].values / hic_bin_size) - start_hic).astype(int)
        #             hic_score = hd["score"].values
        #             # drop contacts with regions outside the [start_row : end_row] range
        #             lix = (l2 < len(hic_mat)) & (l2 >= 0) & (l1 >= 0)
        #             l1 = l1[lix]
        #             l2 = l2[lix]
        #             hic_score = hic_score[lix]
        #             hic_mat[l1, l2] += hic_score
        #             # hic_mat = hic_mat + hic_mat.T - np.diag(np.diag(hic_mat))
        #             hic_mat = gaussian_filter(hic_mat, sigma=1)
        #             # print(f"original {len(hic_mat.flatten())}")
        #             hic_mat = hic_mat[np.triu_indices_from(hic_mat, k=1)]
        #             # print(f"triu {len(hic_mat.flatten())}")
        #             for hs in range(hic_track_size):
        #                 hic_slice = hic_mat[hs * num_regions: (hs + 1) * num_regions].copy()
        #                 if len(hic_slice) != num_regions:
        #                     hic_slice.resize(num_regions, refcheck=False)
        #                 output_scores[ni].append(hic_slice)
        #             ni += 1
        #         except Exception as e:
        #             print(e)
        #             err += 1
        #     del hd
        #     del hdf
        #     gc.collect()
        # print(np.asarray(input_sequences).shape)
        # print(np.asarray(output_scores).shape)
        print("\nTest output")
        # preparing test output tracks
        test_output = []
        for i in range(len(test_info)):
            scores = []
            start = int((test_info[i][1] - half_size) / bin_size)
            for key in chosen_tracks:
                scores.append(gas[key][test_info[i][0]][start: start + num_regions])
            test_output.append(scores)
        # for key in hic_keys:
        #     print(key, end=" ")
        #     hdf = joblib.load("parsed_hic/" + key)
        #     for i, info in enumerate(test_info):
        #         try:
        #             hd = hdf[info[0]]
        #             hic_mat = np.zeros((num_hic_bins, num_hic_bins))
        #             start_hic = int((info[1] - half_size))
        #             end_hic = start_hic + input_size
        #             start_row = hd['locus1'].searchsorted(start_hic - hic_bin_size, side='left')
        #             end_row = hd['locus1'].searchsorted(end_hic, side='right')
        #             hd = hd.iloc[start_row:end_row]
        #             # convert start of the input region to the bin number
        #             start_hic = int(start_hic / hic_bin_size)
        #             # subtract start bin from the binned entries in the range [start_row : end_row]
        #             l1 = (np.floor(hd["locus1"].values / hic_bin_size) - start_hic).astype(int)
        #             l2 = (np.floor(hd["locus2"].values / hic_bin_size) - start_hic).astype(int)
        #             hic_score = hd["score"].values
        #             # drop contacts with regions outside the [start_row : end_row] range
        #             lix = (l2 < len(hic_mat)) & (l2 >= 0) & (l1 >= 0)
        #             l1 = l1[lix]
        #             l2 = l2[lix]
        #             hic_score = hic_score[lix]
        #             hic_mat[l1, l2] += hic_score
        #             # hic_mat = hic_mat + hic_mat.T - np.diag(np.diag(hic_mat))
        #             hic_mat = gaussian_filter(hic_mat, sigma=1)
        #             # print(f"original {len(hic_mat.flatten())}")
        #             hic_mat = hic_mat[np.triu_indices_from(hic_mat, k=1)]
        #             # print(f"triu {len(hic_mat.flatten())}")
        #             for hs in range(hic_track_size):
        #                 hic_slice = hic_mat[hs * num_regions: (hs + 1) * num_regions].copy()
        #                 if len(hic_slice) != num_regions:
        #                     hic_slice.resize(num_regions, refcheck=False)
        #                 test_output[i].append(hic_slice)
        #         except Exception as e:
        #             print(e)
        #             err += 1
        #     del hd
        #     del hdf
        #     gc.collect()

        test_output = np.asarray(test_output).astype(np.float16)
        # print(test_output.shape)

        print("")
        print(datetime.now().strftime('[%H:%M:%S] ') + "Problems: " + str(err))
        output_scores = np.asarray(output_scores).astype(np.float16)
        input_sequences = np.asarray(input_sequences)
        gc.collect()
        print_memory()
        for name, size in sorted(((name, sys.getsizeof(value)) for name, value in locals().items()),
                                 key=lambda x: -x[1])[:10]:
            print("{:>30}: {:>8}".format(name, cm.get_human_readable(size)))
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
                fit_epochs = 6
            else:
                fit_epochs = 2
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
            optimizer = tfa.optimizers.AdamW(
                learning_rate=lr, weight_decay=0.0001
            )
            our_model.compile(loss="mse", optimizer=optimizer)

        print(datetime.now().strftime('[%H:%M:%S] ') + "Training")
        gc.collect()
        print_memory()
        # if k != 0:
        try:
            train_data = wrap(input_sequences, output_scores, GLOBAL_BATCH_SIZE)
            gc.collect()
            our_model.fit(train_data, epochs=fit_epochs)
            our_model.save(model_folder + "/" + model_name + "_temp.h5")
            os.remove(model_folder + "/" + model_name)
            os.rename(model_folder + "/" + model_name + "_temp.h5", model_folder + "/" + model_name)
            for i, key in enumerate(chosen_tracks):
                joblib.dump(our_model.get_layer("out_row_" + str(i)).get_weights(), model_folder + "/" + key + "_temp",
                            compress=3)
                if os.path.exists(model_folder + "/" + key):
                    os.remove(model_folder + "/" + key)
                os.rename(model_folder + "/" + key + "_temp", model_folder + "/" + key)
            if k == 0 and model_was_created:
                all_weights = []
                for i in range(len(chosen_tracks)):
                    all_weights.append(our_model.get_layer("out_row_" + str(i)).get_weights()[0])
                all_weights = np.asarray(all_weights)
                all_weights = np.mean(all_weights, axis=0)
                joblib.dump([all_weights], model_folder + "/avg", compress=3)
                for i in range(len(gas_keys)):
                    if gas_keys[i] in chosen_tracks:
                        continue
                    shutil.copyfile(model_folder + "/avg", model_folder + "/" + gas_keys[i])
            del train_data
            gc.collect()
        except Exception as e:
            print(e)
            print(datetime.now().strftime('[%H:%M:%S] ') + "Error while training. Loading previous model.")
            K.clear_session()
            tf.compat.v1.reset_default_graph()
            gc.collect()
            time.sleep(5)
            with strategy.scope():
                our_model = tf.keras.models.load_model(model_folder + "/" + model_name,
                                                       custom_objects={'PatchEncoder': mo.PatchEncoder})
            input_sequences = None
            output_scores = None
            predictions = None
            gc.collect()
            continue

        if k % 5 == 0 : # and k != 0
            print(datetime.now().strftime('[%H:%M:%S] ') + "Evaluating")
            try:
                print("Training set")
                predictions = our_model.predict(input_sequences[0:1000], batch_size=1)

                corrs = {}
                for it, ct in enumerate(chosen_tracks):
                    type = ct[ct.find("tracks_")+len("tracks_"):ct.find(".")]
                    a = []
                    b = []
                    for i in range(len(predictions)):
                        a.append(predictions[i][it][mid_bin])
                        b.append(output_scores[i][it][mid_bin])
                    corrs.setdefault(type, []).append(stats.spearmanr(a, b)[0])
                for track_type in corrs.keys():
                    print(f"{track_type} correlation : {np.mean(corrs[track_type])}")

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
                # print("Drawing contact maps")
                # for h in range(len(hic_keys)):
                #     pic_count = 0
                #     it = len(chosen_tracks) + h * hic_track_size
                #     for i in range(len(predictions)):
                #         mat_gt = recover_shape(output_scores[i][it:it + hic_track_size], num_hic_bins)
                #         mat_pred = recover_shape(predictions[i][it:it + hic_track_size], num_hic_bins)
                #         fig, axs = plt.subplots(2, 1, figsize=(6, 8))
                #         sns.heatmap(mat_pred, linewidth=0.0, ax=axs[0])
                #         axs[0].set_title("Prediction")
                #         sns.heatmap(mat_gt, linewidth=0.0, ax=axs[1])
                #         axs[1].set_title("Ground truth")
                #         plt.tight_layout()
                #         plt.savefig(figures_folder + "/hic/train_track_" + str(i + 1) + "_" + str(hic_keys[h]) + ".png")
                #         plt.close(fig)
                #         pic_count += 1
                #         if pic_count > 5:
                #             break

                print("Test set")
                predictions = our_model.predict(test_seq[0:1000], batch_size=1)

                corrs = {}
                for it, ct in enumerate(chosen_tracks):
                    type = ct[ct.find("tracks_") + len("tracks_"):ct.find(".")]
                    a = []
                    b = []
                    for i in range(len(predictions)):
                        a.append(predictions[i][it][mid_bin])
                        b.append(test_output[i][it][mid_bin])
                    corrs.setdefault(type, []).append(stats.spearmanr(a, b)[0])
                for track_type in corrs.keys():
                    print(f"{track_type} correlation : {np.mean(corrs[track_type])}")

                with open("result.txt", "a+") as myfile:
                    myfile.write(datetime.now().strftime('[%H:%M:%S] ') + "\n")
                    for track_type in corrs.keys():
                        myfile.write(str(track_type) + "\t")
                    for track_type in corrs.keys():
                        myfile.write(str(np.mean(corrs[track_type])) + "\t")
                    myfile.write("\n")

                print("Drawing tracks")
                pic_count = 0
                for it, ct in enumerate(chosen_tracks):
                    for i in range(len(predictions)):
                        if np.sum(test_output[i][it]) == 0:
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

                # print("Drawing contact maps")
                # for h in range(len(hic_keys)):
                #     pic_count = 0
                #     it = len(chosen_tracks) + h * hic_track_size
                #     for i in range(len(predictions)):
                #         mat_gt = recover_shape(test_output[i][it: it + hic_track_size], num_hic_bins)
                #         mat_pred = recover_shape(predictions[i][it: it + hic_track_size], num_hic_bins)
                #         fig, axs = plt.subplots(2, 1, figsize=(6, 8))
                #         sns.heatmap(mat_pred, linewidth=0.0, ax=axs[0])
                #         axs[0].set_title("Prediction")
                #         sns.heatmap(mat_gt, linewidth=0.0, ax=axs[1])
                #         axs[1].set_title("Ground truth")
                #         plt.tight_layout()
                #         plt.savefig(figures_folder + "/hic/test_track_" + str(i + 1) + "_" + str(hic_keys[h]) + ".png")
                #         plt.close(fig)
                #         pic_count += 1
                #         if pic_count > 5:
                #             break

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
                predictions = None
            except Exception as e:
                print(e)
                print(datetime.now().strftime('[%H:%M:%S] ') + "Problem during evaluation")
        print_memory()
        print(datetime.now().strftime('[%H:%M:%S] ') + "Cleaning")
        # Needed to prevent Keras memory leak
        input_sequences = None
        output_scores = None
        test_output = None
        our_model = None
        gas = None
        gc.collect()
        K.clear_session()
        tf.compat.v1.reset_default_graph()
        print_memory()
        for name, size in sorted(((name, sys.getsizeof(value)) for name, value in locals().items()),
                                 key=lambda x: -x[1])[:10]:
            print("{:>30}: {:>8}".format(name, cm.get_human_readable(size)))
        print(datetime.now().strftime('[%H:%M:%S] ') + "Epoch " + str(k) + " finished. ")


def print_memory():
    mem = psutil.virtual_memory()
    print(f"used: {cm.get_human_readable(mem.used)} available: {cm.get_human_readable(mem.available)}")


def recover_shape(v, size_X):
    v = v.flatten()
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
    # get the current folder absolute path
    # os.chdir(open("data_dir").read().strip())
    os.chdir("/home/acd13586qv/variants")
    train()
