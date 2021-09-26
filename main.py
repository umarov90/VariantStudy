import os

import joblib

# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = '2'
import logging

# logging.getLogger("tensorflow").setLevel(logging.ERROR)
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
from scipy import stats
from multiprocessing import Pool
import shutil
import psutil
import sys
import parse_data as parser
from datetime import datetime
import traceback
import multiprocessing as mp
import pickle
from heapq import nsmallest
import copy
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
matplotlib.use("agg")
# from scipy.ndimage.filters import gaussian_filter
# from sam import SAM

# tf.compat.v1.disable_eager_execution()
# physical_devices = tf.config.experimental.list_physical_devices('GPU')
# assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
# for device in physical_devices:
#     config1 = tf.config.experimental.set_memory_growth(device, True)

# seed = 0
# random.seed(seed)
# np.random.seed(seed)
# tf.random.set_seed(seed)
model_folder = "model_6010"
model_name = "expression_model_1.h5"
figures_folder = "figures_1"
input_size = 120100
half_size = int(input_size / 2)
bin_size = 100
max_shift = 0
hic_bin_size = 10000
num_hic_bins = int(input_size / hic_bin_size)
num_bins = 801  # int(input_size / bin_size)
half_num_bins = 400
mid_bin = math.floor(num_bins / 2)
BATCH_SIZE = 4
out_stack_num = 6010
STEPS_PER_EPOCH = 200
chromosomes = ["chrX"]  # "chrY"
for i in range(1, 23):
    chromosomes.append("chr" + str(i))
num_epochs = 10000
hic_track_size = 1


def recompile(q):
    import tensorflow as tf
    strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy()
    with strategy.scope():
        our_model = tf.keras.models.load_model(model_folder + "/" + model_name,
                                               custom_objects={'SAMModel': mo.SAMModel,
                                                               'PatchEncoder': mo.PatchEncoder})
        print(datetime.now().strftime('[%H:%M:%S] ') + "Compiling model")
        lr = 0.0005
        with strategy.scope():
            optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
            our_model.compile(loss="mse", optimizer=optimizer)

        our_model.save(model_folder + "/" + model_name)
        print("Model saved " + model_folder + "/" + model_name)
    q.put(None)


def create_model(q):
    import tensorflow as tf
    strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy()
    with strategy.scope():
        our_model = mo.simple_model(input_size, num_bins, out_stack_num)
        print(datetime.now().strftime('[%H:%M:%S] ') + "Compiling model")
        lr = 0.0001
        with strategy.scope():
            # base_optimizer = tf.keras.optimizers.SGD(learning_rate=lr, momentum=0.9)
            # base_optimizer = LossScaleOptimizer(base_optimizer, initial_scale=2 ** 2)
            # optimizer = SAM(base_optimizer)
            optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
            our_model.compile(loss="mse", optimizer=optimizer)

        Path(model_folder).mkdir(parents=True, exist_ok=True)
        our_model.save(model_folder + "/" + model_name)
        print("Model saved " + model_folder + "/" + model_name)
    q.put(None)


def run_epoch(q, k, train_info, test_info, one_hot, track_names, eval_track_names):
    import tensorflow as tf
    from tensorflow.keras import mixed_precision
    mixed_precision.set_global_policy('mixed_float16')
    strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy()

    def wrap(input_sequences, output_scores, bs):
        train_data = tf.data.Dataset.from_tensor_slices((input_sequences, output_scores))
        train_data = train_data.batch(bs)
        options = tf.data.Options()
        options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA
        train_data = train_data.with_options(options)
        return train_data

    def wrap2(input_sequences, bs):
        train_data = tf.data.Dataset.from_tensor_slices(input_sequences)
        train_data = train_data.batch(bs)
        options = tf.data.Options()
        options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA
        train_data = train_data.with_options(options)
        return train_data

    GLOBAL_BATCH_SIZE = BATCH_SIZE * strategy.num_replicas_in_sync

    print(datetime.now().strftime('[%H:%M:%S] ') + "Epoch " + str(k))
    # rng_state = np.random.get_state()
    # np.random.shuffle(input_sequences)
    # np.random.set_state(rng_state)
    # np.random.shuffle(output_scores)
    random.shuffle(train_info)

    input_sequences = []
    output_scores = []
    print(datetime.now().strftime('[%H:%M:%S] ') + "Loading the model")
    with strategy.scope():
        our_model = tf.keras.models.load_model(model_folder + "/" + model_name,
                                               custom_objects={'SAMModel': mo.SAMModel,
                                                               'PatchEncoder': mo.PatchEncoder})
    # our_model.get_layer("last_conv1d").set_weights(joblib.load(model_folder + "/head" + str(head_id)))

    print(datetime.now().strftime('[%H:%M:%S] ') + "Preparing sequences")
    err = 0
    for i, info in enumerate(train_info):
        if len(input_sequences) >= GLOBAL_BATCH_SIZE * STEPS_PER_EPOCH:
            break
        if info[0] not in chromosomes:
            continue
        if i % 500 == 0:
            print(i, end=" ")
            gc.collect()
        try:
            rand_var = random.randint(0, max_shift)
            start = int(info[1] + (rand_var - max_shift / 2) - half_size)
            extra = start + input_size - len(one_hot[info[0]])
            if start < 0 or extra > 0:
                continue
            if start < 0:
                ns = one_hot[info[0]][0:start + input_size]
                ns = np.concatenate((np.zeros((-1*start, 4)), ns))
            elif extra > 0:
                ns = one_hot[info[0]][start: len(one_hot[info[0]])]
                ns = np.concatenate((ns, np.zeros((extra, 4))))
            else:
                ns = one_hot[info[0]][start:start + input_size]
            input_sequences.append(ns)
            out_arr = joblib.load("parsed_data_processed/" + info[-1] + ".gz")
            output_scores.append(out_arr)
        except Exception as e:
            print(e)
            err += 1
    print("")
    print(datetime.now().strftime('[%H:%M:%S] ') + "Problems: " + str(err))
    gc.collect()
    print_memory()
    output_scores = np.asarray(output_scores, dtype=np.float16)
    input_sequences = np.asarray(input_sequences, dtype=np.float16)
    gc.collect()
    print_memory()
    for name, size in sorted(((name, sys.getsizeof(value)) for name, value in locals().items()),
                             key=lambda x: -x[1])[:10]:
        print("{:>30}: {:>8}".format(name, cm.get_human_readable(size)))


    print(datetime.now().strftime('[%H:%M:%S] ') + "Training")
    gc.collect()
    print_memory()

    # if k != 0:
    try:
        fit_epochs = 2
        train_data = wrap(input_sequences, output_scores, GLOBAL_BATCH_SIZE)
        gc.collect()
        our_model.fit(train_data, epochs=fit_epochs)
        our_model.save(model_folder + "/" + model_name + "_temp.h5")
        os.remove(model_folder + "/" + model_name)
        os.rename(model_folder + "/" + model_name + "_temp.h5", model_folder + "/" + model_name)
        # joblib.dump(our_model.get_layer("last_conv1d").get_weights(),
        #             model_folder + "/head" + str(head_id) + "_temp", compress=3)
        # if os.path.exists(model_folder + "/head" + str(head_id)):
        #     os.remove(model_folder + "/head" + str(head_id))
        # os.rename(model_folder + "/head" + str(head_id) + "_temp", model_folder + "/head" + str(head_id))
        del train_data
        gc.collect()
    except Exception as e:
        print(e)
        print(datetime.now().strftime('[%H:%M:%S] ') + "Error while training.")
        q.put(None)
        return None

    if k % 5 == 0:  # and k != 0
        print(datetime.now().strftime('[%H:%M:%S] ') + "Evaluating")
        try:
            # print("Training set")
            # predictions = our_model.predict(wrap2(input_sequences[0:1000], GLOBAL_BATCH_SIZE))
            #
            # corrs = {}
            # for it, ct in enumerate(chosen_tracks):
            #     type = ct[ct.find("tracks_") + len("tracks_"):ct.find(".")]
            #     a = []
            #     b = []
            #     for i in range(len(predictions)):
            #         a.append(predictions[i][it][mid_bin])
            #         b.append(output_scores[i][it][mid_bin])
            #     corrs.setdefault(type, []).append((stats.spearmanr(a, b)[0], ct))
            #
            # for track_type in corrs.keys():
            #     print(f"{track_type} correlation : {np.mean([i[0] for i in corrs[track_type]])}")
            #
            # with open("result_cage_train.csv", "w+") as myfile:
            #     for ccc in corrs["CAGE"]:
            #         myfile.write(str(ccc[0]) + "," + str(ccc[1]))
            #         myfile.write("\n")
            #
            # print("Across all genes corrs")
            # corrs = {}
            # a = {}
            # b = {}
            # for gene in range(len(predictions)):
            #     for it, track in enumerate(chosen_tracks):
            #         type = track[track.find("tracks_") + len("tracks_"):track.find(".")]
            #         if type != "CAGE":
            #             continue
            #         if track not in eval_tracks:
            #             continue
            #         a.setdefault(gene, []).append(predictions[gene][it][mid_bin])
            #         b.setdefault(gene, []).append(output_scores[gene][it][mid_bin])
            # a1 = []
            # b1 = []
            # for key in a.keys():
            #     pred_mean = np.mean(a[key])
            #     gt_mean = np.mean(b[key])
            #     a1.append(pred_mean)
            #     b1.append(gt_mean)
            #
            # acorr = stats.spearmanr(a1, b1)[0]
            # print(f"Across all genes spearmanr {acorr}")

            #
            # print("Drawing tracks")
            #
            # total_pics = 0
            # for it, ct in enumerate(chosen_tracks):
            #     if "ctss" not in ct:
            #         continue
            #     pic_count = 0
            #     r = list(range(len(predictions)))
            #     random.shuffle(r)
            #     for i in r:
            #         if np.sum(output_scores[i][it]) == 0:
            #             continue
            #         fig, axs = plt.subplots(2, 1, figsize=(12, 8))
            #         vector1 = predictions[i][it]
            #         vector2 = output_scores[i][it]
            #         x = range(num_regions)
            #         d1 = {'bin': x, 'expression': vector1}
            #         df1 = pd.DataFrame(d1)
            #         d2 = {'bin': x, 'expression': vector2}
            #         df2 = pd.DataFrame(d2)
            #         sns.lineplot(data=df1, x='bin', y='expression', ax=axs[0])
            #         axs[0].set_title("Prediction")
            #         sns.lineplot(data=df2, x='bin', y='expression', ax=axs[1])
            #         axs[1].set_title("Ground truth")
            #         fig.tight_layout()
            #         plt.savefig(figures_folder + "/tracks/train_track_" + str(i + 1) + "_" + str(ct) + ".png")
            #         plt.close(fig)
            #         pic_count += 1
            #         total_pics += 1
            #         if pic_count > 10:
            #             break
            #     if total_pics > 100:
            #         break

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
            # input_sequences = None
            # output_scores = None
            # train_data = None
            # gc.collect()
            print("Test set")

            test_output = []
            for i, info in enumerate(test_info):
                out_arr = joblib.load("parsed_data_processed/" + info[-1] + ".gz")
                test_output.append(out_arr)
            test_output = np.asarray(test_output, dtype=np.float16)

            final_test_pred = {}
            for i in range(len(test_info)):
                final_test_pred[test_info[i][2]] = {}

            for shift_val in [0]:  # -2 * bin_size, -1 * bin_size, 0, bin_size, 2 * bin_size
                test_seq = []
                for info in test_info:
                    start = int(info[1] + shift_val - half_size)
                    extra = start + input_size - len(one_hot[info[0]])
                    if start < 0:
                        ns = one_hot[info[0]][0:start + input_size]
                        ns = np.concatenate((np.zeros((-1*start, 4)), ns))
                    elif extra > 0:
                        ns = one_hot[info[0]][start: len(one_hot[info[0]])]
                        ns = np.concatenate((ns, np.zeros((extra, 4))))
                    else:
                        ns = one_hot[info[0]][start:start + input_size]
                    if len(ns) != input_size:
                        print(f"Wrong! {ns.shape} {start} {extra} {info[1]}")
                    test_seq.append(ns)
                for comp in [False]:
                    if comp:
                        with Pool(4) as p:
                            rc_arr = p.map(change_seq, test_seq)
                        test_seq = rc_arr
                    test_seq = np.asarray(test_seq, dtype=np.float16)
                    if comp:
                        correction = 1 * int(shift_val / bin_size)
                    else:
                        correction = -1 * int(shift_val / bin_size)
                    print(f"\n{shift_val} {comp} {test_seq.shape} predicting")
                    predictions = None
                    for w in range(0, len(test_seq), 1000):
                        print(w, end=" ")
                        gc.collect()
                        if w == 0:
                            predictions = our_model.predict(wrap2(test_seq[w:w+1000], GLOBAL_BATCH_SIZE))[:, :, mid_bin + correction]
                        else:
                            new_predictions = our_model.predict(wrap2(test_seq[w:w + 1000], GLOBAL_BATCH_SIZE))[:, :, mid_bin + correction]
                            predictions = np.concatenate((predictions, new_predictions), dtype=np.float16)
                    for i in range(len(test_info)):
                        for it, ct in enumerate(track_names):
                            final_test_pred[test_info[i][2]].setdefault(ct, []).append(predictions[i][it])
                    print(f"{shift_val} {comp} finished")
                    predictions = None
                    gc.collect()

            for i, gene in enumerate(final_test_pred.keys()):
                if i % 10 == 0:
                    print(i, end=" ")
                for track in track_names:
                    final_test_pred[gene][track] = np.mean(final_test_pred[gene][track])

            corr_p = []
            corr_s = []
            for gene in final_test_pred.keys():
                a = []
                b = []
                for track in track_names:
                    type = track[track.find("tracks_") + len("tracks_"):track.find(".")]
                    if type != "CAGE":
                        continue
                    if track not in eval_tracks:
                        continue
                    a.append(final_test_pred[gene][track])
                    b.append(test_output[gene][track])
                pc = stats.pearsonr(a, b)[0]
                sc = stats.spearmanr(a, b)[0]
                if not math.isnan(sc) and not math.isnan(pc):
                    corr_p.append(pc)
                    corr_s.append(sc)
            # a1 = []
            # b1 = []
            # for key in a.keys():
            #     pred_mean = np.mean(a[key])
            #     gt_mean = np.mean(b[key])
            #     a1.append(pred_mean)
            #     b1.append(gt_mean)

            print(f"Maybe this 44444444444444 {len(corr_p)} {np.mean(corr_p)} {np.mean(corr_s)}")

            # a = {}
            # b = {}
            # for gene in final_test_pred.keys():
            #     if gene not in protein_coding:
            #         continue
            #     for track in gas_keys:
            #         type = track[track.find("tracks_") + len("tracks_"):track.find(".")]
            #         if type != "CAGE":
            #             continue
            #         if track not in eval_tracks:
            #             continue
            #         a.setdefault(gene, []).append(final_test_pred[gene][track])
            #         b.setdefault(gene, []).append(test_output[gene][track])
            # a1 = []
            # b1 = []
            # for key in a.keys():
            #     pred_mean = np.mean(a[key])
            #     gt_mean = np.mean(b[key])
            #     a1.append(pred_mean)
            #     b1.append(gt_mean)
            #
            # print(f"Across all genes eval tracks protein coding {stats.pearsonr(a1, b1)[0]} {stats.spearmanr(a1, b1)[0]}")
            #
            # a = {}
            # b = {}
            # for gene in final_test_pred.keys():
            #     if gene not in protein_coding:
            #         continue
            #     for track in gas_keys:
            #         type = track[track.find("tracks_") + len("tracks_"):track.find(".")]
            #         if type != "CAGE":
            #             continue
            #         if track not in eval_tracks:
            #             continue
            #         a.setdefault(gene, []).append(final_test_pred[gene][track])
            #         b.setdefault(gene, []).append(test_output[gene][track])
            # a1 = []
            # b1 = []
            # for key in a.keys():
            #     pred_mean = np.mean(a[key])
            #     gt_mean = np.mean(b[key])
            #     a1.append(pred_mean)
            #     b1.append(gt_mean)
            #
            # print(
            #     f"Across all genes eval tracks protein coding {stats.pearsonr(a1, b1)[0]} {stats.spearmanr(a1, b1)[0]}")
            #
            # a = {}
            # b = {}
            # for gene in final_test_pred.keys():
            #     if gene not in protein_coding:
            #         continue
            #     for track in gas_keys:
            #         type = track[track.find("tracks_") + len("tracks_"):track.find(".")]
            #         if type != "CAGE":
            #             continue
            #         a.setdefault(gene, []).append(final_test_pred[gene][track])
            #         b.setdefault(gene, []).append(test_output[gene][track])
            # a1 = []
            # b1 = []
            # for key in a.keys():
            #     pred_mean = np.mean(a[key])
            #     gt_mean = np.mean(b[key])
            #     a1.append(pred_mean)
            #     b1.append(gt_mean)
            #
            # print(f"Across all genes all tracks protein coding {stats.pearsonr(a1, b1)[0]} {stats.spearmanr(a1, b1)[0]}")

            # print("Accross tracks")
            # corrs_p = {}
            # corrs_s = {}
            # for track in gas_keys:
            #     type = track[track.find("tracks_") + len("tracks_"):track.find(".")]
            #     a = []
            #     b = []
            #     for gene in final_test_pred.keys():
            #         a.append(final_test_pred[gene][track])
            #         b.append(test_output[gene][track])
            #     corrs_p.setdefault(type, []).append((stats.pearsonr(a, b)[0], track))
            #     corrs_s.setdefault(type, []).append((stats.spearmanr(a, b)[0], track))
            #
            # for track_type in corrs_p.keys():
            #     print(f"{track_type} correlation : {np.mean([i[0] for i in corrs_p[track_type]])} {np.mean([i[0] for i in corrs_s[track_type]])}")

            print("Accross tracks protein coding")
            corrs_p = {}
            corrs_s = {}
            for track in track_names:
                type = track[track.find("tracks_") + len("tracks_"):track.find(".")]
                a = []
                b = []
                for gene in final_test_pred.keys():
                    a.append(final_test_pred[gene][track])
                    b.append(test_output[gene][track])
                corrs_p.setdefault(type, []).append((stats.pearsonr(a, b)[0], track))
                corrs_s.setdefault(type, []).append((stats.spearmanr(a, b)[0], track))

            for track_type in corrs_p.keys():
                print(
                    f"{track_type} correlation : {np.mean([i[0] for i in corrs_p[track_type]])} {np.mean([i[0] for i in corrs_s[track_type]])}")

            with open("result_cage_test.csv", "w+") as myfile:
                for ccc in corrs_p["CAGE"]:
                    myfile.write(str(ccc[0]) + "," + str(ccc[1]))
                    myfile.write("\n")

            with open("result.txt", "a+") as myfile:
                myfile.write(datetime.now().strftime('[%H:%M:%S] ') + "\n")
                for track_type in corrs_p.keys():
                    myfile.write(str(track_type) + "\t")
                for track_type in corrs_p.keys():
                    myfile.write(str(np.mean([i[0] for i in corrs_p[track_type]])) + "\t")
                myfile.write("\n")

            # print("Drawing tracks")
            # pic_count = 0
            # for it, ct in enumerate(chosen_tracks):
            #     if "ctss" not in ct:
            #         continue
            #     for i in range(len(predictions)):
            #         if np.sum(test_output[i][it]) == 0:
            #             continue
            #         fig, axs = plt.subplots(2, 1, figsize=(12, 8))
            #         vector1 = predictions[i][it]
            #         vector2 = test_output[i][it]
            #         x = range(num_regions)
            #         d1 = {'bin': x, 'expression': vector1}
            #         df1 = pd.DataFrame(d1)
            #         d2 = {'bin': x, 'expression': vector2}
            #         df2 = pd.DataFrame(d2)
            #         sns.lineplot(data=df1, x='bin', y='expression', ax=axs[0])
            #         axs[0].set_title("Prediction")
            #         sns.lineplot(data=df2, x='bin', y='expression', ax=axs[1])
            #         axs[1].set_title("Ground truth")
            #         fig.tight_layout()
            #         plt.savefig(
            #             figures_folder + "/tracks/test_track_" + str(testinfo_small[i][2]) + "_" + str(ct) + ".png")
            #         plt.close(fig)
            #         pic_count += 1
            #         if i > 20:
            #             break
            #     if pic_count > 100:
            #         break

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
        except Exception as e:
            print(e)
            traceback.print_exc()
            print(datetime.now().strftime('[%H:%M:%S] ') + "Problem during evaluation")
    print(datetime.now().strftime('[%H:%M:%S] ') + "Epoch " + str(k) + " finished. ")
    q.put(None)

def print_memory():
    mem = psutil.virtual_memory()
    print(f"used: {cm.get_human_readable(mem.used)} available: {cm.get_human_readable(mem.available)}")


def recover_shape(v, size_X):
    v = v.flatten()
    end = int((size_X * size_X - size_X) / 2)
    v = v[:end]
    X = np.zeros((size_X, size_X))
    X[np.triu_indices(X.shape[0], k=1)] = v
    X = X + X.T
    return X


def change_seq(x):
    return cm.rev_comp(x)


if __name__ == '__main__':
    # get the current folder absolute path
    # os.chdir(open("data_dir").read().strip())
    os.chdir("/home/acd13586qv/variants")
    # How does enformer handle strands???
    # read training notebook
    # our_model = mo.simple_model(input_size, num_regions, out_stack_num)
    Path(model_folder).mkdir(parents=True, exist_ok=True)
    Path(figures_folder + "/" + "attribution").mkdir(parents=True, exist_ok=True)
    Path(figures_folder + "/" + "tracks").mkdir(parents=True, exist_ok=True)
    Path(figures_folder + "/" + "hic").mkdir(parents=True, exist_ok=True)

    one_hot, train_info, test_info = parser.get_sequences(chromosomes, input_size)
    if Path("pickle/track_names.gz").is_file():
        track_names = joblib.load("pickle/track_names.gz")
    else:
        track_names = parser.parse_tracks(train_info, test_info, bin_size, half_num_bins)

    print("Number of tracks: " + str(len(track_names)))

    print_memory()
    for name, size in sorted(((name, sys.getsizeof(value)) for name, value in locals().items()),
                             key=lambda x: -x[1])[:10]:
        print("{:>30}: {:>8}".format(name, cm.get_human_readable(size)))

    if Path("pickle/eval_gas.gz").is_file():
        eval_track_names = joblib.load("pickle/eval_track_names.gz")
    else:
        eval_tracks = pd.read_csv('eval_tracks.tsv', delimiter='\t').values.flatten()
        eval_track_names = []
        for tn in track_names:
            found = False
            for i in range(len(eval_tracks)):
                if eval_tracks[i] in tn:
                    found = True
                    break
            if found:
                eval_track_names.append(tn)
        joblib.dump(eval_track_names, "pickle/eval_track_names.gz", compress=3)

    # mp.set_start_method('spawn', force=True)
    # try:
    #     mp.set_start_method('spawn')
    # except RuntimeError:
    #     pass
    q = mp.Queue()
    if not Path(model_folder + "/" + model_name).is_file():
        p = mp.Process(target=create_model, args=(q,))
        p.start()
        print(q.get())
        p.join()
        time.sleep(1)
    else:
        print("Model exists")
    # p = mp.Process(target=recompile, args=(q,))
    # p.start()
    # print(q.get())
    # p.join()
    # time.sleep(1)
    print("Training starting")
    for k in range(num_epochs):
        # run_epoch(q, k, train_info, test_info, heads, one_hot,gas_keys,eval_gas)
        p = mp.Process(target=run_epoch, args=(q, k, train_info, test_info, one_hot, track_names, eval_track_names,))
        p.start()
        print(q.get())
        p.join()
        time.sleep(1)
