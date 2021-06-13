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
import skimage.measure
import time
from datetime import datetime
matplotlib.use("agg")


def train():
    # Apply smoothing to output bins? Ask Hon how to do it best
    # transformer_layers = 8 Try 1 instead of 8
    model_folder = "model1"
    model_name = "expression_model_1.h5"
    figures_folder = "figures_1"
    input_size = 100000
    half_size = input_size / 2
    max_shift = 20
    bin_size = 1000
    num_regions = int(input_size / bin_size)
    mid_bin = math.floor(num_regions / 2)
    BATCH_SIZE = 1
    strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy()
    print('Number of devices: {}'.format(strategy.num_replicas_in_sync))
    GLOBAL_BATCH_SIZE = BATCH_SIZE * strategy.num_replicas_in_sync
    STEPS_PER_EPOCH = 3000
    out_stack_num = 1000
    num_epochs = 10000

    Path(figures_folder + "/" + "attribution").mkdir(parents=True, exist_ok=True)
    Path(figures_folder + "/" + "tracks").mkdir(parents=True, exist_ok=True)

    chromosomes = ["chrX", "chrY"]
    # our_model = mo.simple_model(input_size, num_regions, 3000)
    for i in range(2, 23):
        chromosomes.append("chr" + str(i))

    if Path("pickle/genome.p").is_file():
        genome = pickle.load(open("pickle/genome.p", "rb"))
        ga = pickle.load(open("pickle/ga.p", "rb"))
    else:
        genome, ga = cm.parse_genome("hg19.fa", bin_size)
        pickle.dump(genome, open("pickle/genome.p", "wb"), protocol=pickle.HIGHEST_PROTOCOL)
        pickle.dump(ga, open("pickle/ga.p", "wb"), protocol=pickle.HIGHEST_PROTOCOL)

    if Path("pickle/input_sequences_long.p").is_file():
        input_sequences_long = pickle.load(open("pickle/input_sequences_long.p", "rb"))
        test_input_sequences = pickle.load(open("pickle/test_input_sequences.p", "rb"))
        test_output = pickle.load(open("pickle/test_output.p", "rb"))
        test_class = pickle.load(open("pickle/test_class.p", "rb"))
        test_info = pickle.load(open("pickle/test_info.p", "rb"))
        # gas = joblib.load("pickle/gas.gz")
        # kk = list(gas.keys())
        # for key in kk:
        #     gas[key.replace("/", "_")] = gas.pop(key)
        # # for key in gas.keys():
        # #     joblib.dump(gas[key], "parsed_tracks/" + key, compress=3)
        # joblib.dump(kk, "pickle/keys.gz", compress=3)
        # exit()
        # gas_keys = []
        # for filename in os.listdir("parsed_tracks"):
        #     gas_keys.append(filename)
        # gas_keys.remove("IPSC")
        # gas_keys.remove("DMFB")
        # joblib.dump(gas_keys, "pickle/keys.gz", compress=3)
        gas_keys = joblib.load("pickle/keys.gz")
        output_info = pickle.load(open("pickle/output_info.p", "rb"))
        counts = pickle.load(open("pickle/counts.p", "rb"))
        cells = list(sorted(counts.keys()))
    else:
        print("Parsing tracks")
        # gas = joblib.load("pickle/gas.gz")
        gas = {}

        #
        counts = {}
        directory = "count"
        for filename in os.listdir(directory):
            if filename.endswith(".tsv"):
                cell = filename.split(".")[0]
                df = pd.read_csv(os.path.join(directory, filename), sep="\t")
                df['count'] = 1 + df["count"]
                df['count'] = np.log(df['count'])
                df["count"] = df["count"] / df["count"].max()
                d = dict(zip(df["countRegion_ID"], df["count"]))
                counts[cell] = d
                continue
            else:
                continue
        pickle.dump(counts, open("pickle/counts.p", "wb"), protocol=pickle.HIGHEST_PROTOCOL)
        cells = list(sorted(counts.keys()))
        df = pd.read_csv("DMFB_IPSC.CRE.info.tsv", sep="\t")
        enhancers_ids = df.query("classc == 'distal'")['CREID'].to_list()
        promoters_ids = df.query("classc != 'distal'")['CREID'].to_list()
        coding_ids = df.query("classc == 'coding'")['CREID'].to_list()

        ranges_file = "DMFB_IPSC.CRE.coord.bed"
        promoters, enhancers = read_ranges_2(ranges_file, chromosomes, promoters_ids, enhancers_ids)
        test_promoters, test_enhancers = read_ranges_2(ranges_file, ["chr1"], promoters_ids, enhancers_ids)

        for cell in cells:
            gas[cell] = copy.deepcopy(ga)
        # joblib.dump(gas, "pickle/gas.gz", compress=3)

        over = 0
        for chr in chromosomes:
            for p in promoters[chr] + enhancers[chr]:
                for cell in cells:
                    pos = int(p[0] / bin_size)
                    if gas[cell][chr][pos] != 0:
                        over += 1
                    gas[cell][chr][pos] += counts[cell][p[1]]
                    if pos - 1 > 0:
                        gas[cell][chr][pos - 1] += counts[cell][p[1]]
                    if pos + 1 < num_regions:
                        gas[cell][chr][pos + 1] += counts[cell][p[1]]

        for chr in ["chr1"]:
            for p in test_promoters[chr] + test_enhancers[chr]:
                for cell in cells:
                    pos = int(p[0] / bin_size)
                    if gas[cell][chr][pos] != 0:
                        over += 1
                    gas[cell][chr][pos] += counts[cell][p[1]]
                    if pos - 1 > 0:
                        gas[cell][chr][pos - 1] += counts[cell][p[1]]
                    if pos + 1 < num_regions:
                        gas[cell][chr][pos + 1] += counts[cell][p[1]]

        for cell in cells:
            joblib.dump(gas[cell], "parsed_tracks/" + cell, compress=3)
        print("Overlap: " + str(over))
        test_input_sequences = []
        test_output = []
        test_class = []
        test_info = []
        for chr, chr_cres in test_promoters.items():
            for i in range(len(chr_cres)):
                tss = chr_cres[i][0]
                seq = get_seq(genome, chr, tss, input_size)
                if len(seq) != input_size:
                    continue
                test_input_sequences.append(seq)
                start = int((tss - half_size) / bin_size)
                scores = []
                for key in cells:
                    scores.append(gas[key][chr][start: start + num_regions])
                test_output.append(scores)
                if chr_cres[i][1] in coding_ids:
                    test_class.append(1)
                else:
                    test_class.append(0)
                test_info.append(chr_cres[i][1])

        test_input_sequences = np.asarray(test_input_sequences)
        test_output = np.asarray(test_output)
        print("Test set completed")
        input_sequences_long = []
        output_info = []
        for chr, chr_cres in promoters.items():
            for i in range(len(chr_cres)):
                tss = chr_cres[i][0]
                seq = get_seq(genome, chr, tss, input_size + max_shift)
                if len(seq) != input_size + max_shift:
                    continue
                input_sequences_long.append(seq)
                output_info.append([chr, tss])

        input_sequences_long = np.asarray(input_sequences_long)
        print("Training set completed")
        pickle.dump(input_sequences_long, open("pickle/input_sequences_long.p", "wb"), protocol=pickle.HIGHEST_PROTOCOL)
        pickle.dump(test_input_sequences, open("pickle/test_input_sequences.p", "wb"), protocol=pickle.HIGHEST_PROTOCOL)
        pickle.dump(test_output, open("pickle/test_output.p", "wb"), protocol=pickle.HIGHEST_PROTOCOL)
        pickle.dump(test_class, open("pickle/test_class.p", "wb"), protocol=pickle.HIGHEST_PROTOCOL)
        pickle.dump(test_info, open("pickle/test_info.p", "wb"), protocol=pickle.HIGHEST_PROTOCOL)
        pickle.dump(output_info, open("pickle/output_info.p", "wb"), protocol=pickle.HIGHEST_PROTOCOL)
        del counts
        del gas
        gc.collect()

    # gas_keys = []
    # directory = "tracks"
    # for filename in os.listdir(directory):
    #     if filename.endswith(".gz"):
    #         start = time.time()
    #         fn = os.path.join(directory, filename)
    #         t_name = fn.replace("/", "_")
    #         gas_keys.append(t_name)
    #         gast = copy.deepcopy(ga)
    #         df = pd.read_csv(fn, sep="\t", names=["chr", "start", "end", "m", "score", "strand"], header=None, index_col=False)
    #         chrd = list(df["chr"].unique())
    #         df["mid"] = (df["start"] + (df["end"] - df["start"]) / 2) / bin_size
    #         df = df.astype({"mid": int})
    #
    #         # group the scores over `key` and gather them in a list
    #         grouped_scores = df.groupby("chr").agg(list)
    #
    #         # for each key, value in the dictionary...
    #         for key, val in gast.items():
    #             if key not in chrd:
    #                 continue
    #             # first lookup the positions to update and the corresponding scores
    #             pos, score = grouped_scores.loc[key, ["mid", "score"]]
    #             # fancy indexing
    #             gast[key][pos] += score
    #
    #         max_val = -1
    #         for key in gast.keys():
    #             gast[key] = np.log(gast[key] + 1)
    #             max_val = max(np.max(gast[key]), max_val)
    #         for key in gast.keys():
    #             gast[key] = gast[key] / max_val
    #         joblib.dump(gast, "parsed_tracks/" + t_name, compress=3)
    #         end = time.time()
    #         print("Parsed " + fn + ". Elapsed time: " + str(end - start) + ". Max value: " + str(max_val))
    #
    # # tf tracks #############################################
    # our_model = mo.simple_model(input_size, num_regions, 100)
    # dfa = pd.read_csv("tf_tracks.bed.gz", sep="\t", names=["chr", "start", "end", "m", "score", "strand"],
    #                   header=None,
    #                   index_col=False)
    #
    # # dfa["m"] = dfa["m"].apply(lambda x: x[x.find('.') + 1:]).copy()
    #
    #
    # dfa["mid"] = (dfa["start"] + (dfa["end"] - dfa["start"]) / 2) / bin_size
    # print(dfa["score"].min())
    # dfa['score'] = dfa['score'].replace(0.0, 1.0)
    # print(dfa["score"].min())
    # dfa = dfa.astype({"mid": int})
    # # tf_tracks = list(dfa["m"].unique())
    # # print(len(tf_tracks))
    # # dfa = dfa.groupby('m').filter(lambda x: len(x) <= 30000)
    # tf_tracks = list(dfa["m"].unique())
    # print("After filtering " + str(len(tf_tracks)))
    # for t in tf_tracks:
    #     t_name = "chip_" + t
    #     gast = copy.deepcopy(ga)
    #     if t_name not in gas_keys:
    #         gas_keys.append(t_name)
    #     df = dfa.loc[dfa['m'] == t]
    #     chrd = list(df["chr"].unique())
    #     # group the scores over `key` and gather them in a list
    #     grouped_scores = df.groupby("chr").agg(list)
    #
    #     # for each key, value in the dictionary...
    #     for key, val in gast.items():
    #         if key not in chrd:
    #             continue
    #         # first lookup the positions to update and the corresponding scores
    #         pos, score = grouped_scores.loc[key, ["mid", "score"]]
    #         gast[key][pos] += score
    #
    #     max_val = -1
    #     for key in gast.keys():
    #         gast[key] = np.log(gast[key] + 1)
    #         max_val = max(np.max(gast[key]), max_val)
    #     for key in gast.keys():
    #         gast[key] = gast[key] / max_val
    #
    #     joblib.dump(gast, "parsed_tracks/" + t_name, compress=3)
    #     if not Path(model_folder + "/" + t_name).is_file():
    #         joblib.dump(our_model.get_layer("out_row_0").get_weights(), model_folder + "/" + t_name, compress=3)
    #     print(t_name + " " + str(df.shape[0]) + " " + str(max_val))
    # joblib.dump(gas_keys, "pickle/keys.gz", compress=3)

    ########################################################################
    # hic_keys = []
    # directory = "hic"
    # hic_data = {}
    # for filename in os.listdir(directory):
    #     if filename.endswith(".gz"):
    #         fn = os.path.join(directory, filename)
    #         t_name = fn.replace("/", "_")
    #         hic_keys.append(t_name)
    #         df = pd.read_csv(fn, sep="\t", index_col=False)
    #         df.drop(['relCoverage1', 'relCoverage2', 'relCoverage1',
    #                  'probability', 'expected', 'logObservedOverExpected',
    #                  "locus2_chrom", "locus1_end", "locus2_end"], axis=1, inplace=True)
    #         df.drop(df[df.readCount < 5].index, inplace=True)
    #         df.drop(df[df.qvalue > 0.05].index, inplace=True)
    #         df["score"] = 1.0
    #         # df["score"] = -1 * np.log(df["pvalue"])
    #         # df["score"] = df["score"] / df["score"].max()
    #         df.drop(['readCount', 'qvalue', 'pvalue'], axis=1, inplace=True)
    #         # df.to_csv("parsed_hic/" + t_name,index=False,compression="gzip")
    #         chrd = list(df["locus1_chrom"].unique())
    #         for chr in chrd:
    #             hic_data[t_name + chr] = df.loc[df['locus1_chrom'] == chr].sort_values(by=['locus1_start'])
    #         print(t_name)
    # joblib.dump(hic_data, "pickle/hic_data.gz", compress=3)
    # joblib.dump(hic_keys, "pickle/hic_keys.gz", compress=3)
    hic_data = joblib.load("pickle/hic_data.gz")
    hic_keys = joblib.load("pickle/hic_keys.gz")
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
            for i, key in enumerate(gas_keys):
                joblib.dump(our_model.get_layer("out_row_0").get_weights(), model_folder + "/" + key, compress=3)
                if i == 0:
                    print(our_model.get_layer("out_row_0").get_weights()[0].shape)
                    print(our_model.get_layer("out_row_0").get_weights()[1].shape)
                if i % 50 == 0:
                    print(i, end=" ")
                    gc.collect()
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
    del genome
    del ga
    gc.collect()
    for k in range(num_epochs):
        print("Epoch " + str(k) + datetime.now().strftime(' %H:%M:%S'))
        if k > 0:
            with strategy.scope():
                our_model = tf.keras.models.load_model(model_folder + "/" + model_name,
                                                       custom_objects={'PatchEncoder': mo.PatchEncoder})
        input_sequences = []
        output_scores = []
        print("Preparing sequences" + datetime.now().strftime(' %H:%M:%S'))
        chosen_tracks = random.sample(gas_keys, out_stack_num - len(cells) - len(hic_keys))# - len(hic_keys))
        chip_picks = 0
        for it, ct in enumerate(chosen_tracks):
            if ct.startswith("chip_"):
                chip_picks += 1
        print("Chip tracks: " + str(chip_picks) + datetime.now().strftime(' %H:%M:%S'))
        gas = {}
        for i, key in enumerate(chosen_tracks):
            our_model.get_layer("out_row_" + str(i)).set_weights(joblib.load(model_folder + "/" + key))
            gas[key] = joblib.load("parsed_tracks/" + key)
        for i, cell in enumerate(cells):
            # our_model.get_layer("out_row_" + str(-2 + i)).set_weights(joblib.load(model_folder + "/" + cell))
            gas[cell] = joblib.load("parsed_tracks/" + cell)
        print("Loaded the tracks" + datetime.now().strftime(' %H:%M:%S'))
        err = 0
        for i, seq in enumerate(input_sequences_long):
            if i >= GLOBAL_BATCH_SIZE * STEPS_PER_EPOCH:
                break
            if i % 100 == 0:
                print(i, end=" ")
                gc.collect()
            try:
                rand_var = random.randint(0, max_shift)
                # rand_var = 5
                ns = seq[rand_var: rand_var + input_size, :]
                info = output_info[i]
                start = int((info[1] + (rand_var - max_shift / 2) - half_size) / bin_size)
                scores = []
                for key in chosen_tracks:
                    scores.append(gas[key][info[0]][start: start + num_regions])
                for key in hic_keys:
                    hic_mat = np.zeros((10, 10))
                    # hd = hic_data[key].loc[hic_data[key]['locus1_chrom'] == info[0]]
                    hd = hic_data[key + info[0]]
                    start_hic = int((info[1] + (rand_var - max_shift / 2) - half_size))
                    end_hic = start_hic + input_size
                    start_hic = start_hic - start_hic % 10000
                    start_row = hd['locus1_start'].searchsorted(start_hic, side='left')
                    end_row = hd['locus1_start'].searchsorted(end_hic, side='right')
                    hd = hd.iloc[start_row:end_row]
                    l1 = ((hd["locus1_start"].values - start_hic) / 10000).astype(int)
                    l2 = ((hd["locus2_start"].values - start_hic) / 10000).astype(int)
                    lix = l2 < len(hic_mat)
                    l1 = l1[lix]
                    l2 = l2[lix]
                    hic_mat[l1, l2] += 1 # row["score"]
                    hic_mat = hic_mat + hic_mat.T - np.diag(np.diag(hic_mat))
                    if len(hic_mat.flatten()) != 100:
                        print("ooooooooops   ")
                    scores.append(hic_mat.flatten().astype(np.float32))
                for cell in cells:
                    scores.append(gas[cell][info[0]][start: start + num_regions])
                input_sequences.append(ns)
                output_scores.append(scores)
            except Exception as e:
                print(e)
                err += 1
        print("\nProblems: " + str(err) + datetime.now().strftime(' %H:%M:%S'))
        output_scores = np.asarray(output_scores)
        input_sequences = np.asarray(input_sequences)

        rng_state = np.random.get_state()
        np.random.shuffle(input_sequences)
        np.random.set_state(rng_state)
        np.random.shuffle(output_scores)

        input_sequences = input_sequences[:GLOBAL_BATCH_SIZE * STEPS_PER_EPOCH]
        output_scores = output_scores[:GLOBAL_BATCH_SIZE * STEPS_PER_EPOCH]

        print("Compiling model" + datetime.now().strftime(' %H:%M:%S'))
        # if k < 300:
        #     lr = 0.0001
        # elif k < 600:
        #     lr = 0.00005
        # else:
        #     lr = 0.00002
        lr = 0.0001
        fit_epochs = 1
        with strategy.scope():
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

        # if k != 0:
        print("Training" + datetime.now().strftime(' %H:%M:%S'))
        try:
            our_model.fit(input_sequences, output_scores, epochs=fit_epochs, batch_size=GLOBAL_BATCH_SIZE)
            our_model.save(model_folder + "/" + model_name)
            for i, key in enumerate(chosen_tracks):
                joblib.dump(our_model.get_layer("out_row_" + str(i)).get_weights(), model_folder + "/" + key,
                            compress=3)
        except Exception as e:
            print(e)
            print("Error while training. Loading previous model." + datetime.now().strftime(' %H:%M:%S'))
            with strategy.scope():
                our_model = tf.keras.models.load_model(model_folder + "/" + model_name,
                                                       custom_objects={'PatchEncoder': mo.PatchEncoder})
            del input_sequences
            del output_scores
            del predictions
            gc.collect()

        if k % 10 == 0 : # and k != 0
            print("Training set")
            predictions = our_model.predict(input_sequences[0:1000], batch_size=GLOBAL_BATCH_SIZE)

            for c, cell in enumerate(cells):
                ci = -2 + c
                a = []
                b = []
                for i in range(len(predictions)):
                    # if output_scores[i][c][mid_bin] == 0:
                    #     continue
                    a.append(predictions[i][ci][mid_bin])
                    b.append(output_scores[i][ci][mid_bin])
                corr = stats.spearmanr(a, b)[0]
                print("Correlation " + cell + ": " + str(corr))

            pic_count = 0
            for it, ct in enumerate(chosen_tracks):
                if ct.startswith("chip_"):
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
                        plt.savefig(figures_folder + "/chip/track_" + str(i + 1) + "_" + str(ct) + ".png")
                        plt.close(fig)
                        pic_count += 1
                        break
                if pic_count > 10:
                    break

            for h in range(len(hic_keys)):
                pic_count = 0
                it = len(chosen_tracks) + h
                for i in range(500, 800, 1):
                    if np.sum(output_scores[i][it]) == 0:
                        continue
                    mat_gt = np.reshape(output_scores[i][it], (10,10))
                    mat_pred = np.reshape(predictions[i][it], (10,10))
                    fig, axs = plt.subplots(2, 1, figsize=(8, 8))
                    sns.heatmap(mat_pred, linewidth=0.0, ax=axs[0])
                    axs[0].set_title("Prediction")
                    sns.heatmap(mat_gt, linewidth=0.0, ax=axs[1])
                    axs[1].set_title("Ground truth")
                    plt.tight_layout()
                    plt.savefig(figures_folder + "/hic/track_" + str(i + 1) + "_" + str(hic_keys[h]) + ".png")
                    plt.close(fig)
                    pic_count += 1
                    if pic_count > 4:
                        break


            print("Test set")
            predictions = our_model.predict(test_input_sequences, batch_size=GLOBAL_BATCH_SIZE)
            for c, cell in enumerate(cells):
                ci = -2 + c
                a = []
                b = []
                ap = []
                bp = []
                for i in range(len(predictions)):
                    # if test_output[i][c][mid_bin] == 0:
                    #     continue
                    a.append(predictions[i][ci][mid_bin])
                    b.append(test_output[i][c][mid_bin])
                    if test_class[i] == 0:
                        continue
                    ap.append(predictions[i][ci][mid_bin])
                    bp.append(test_output[i][c][mid_bin])
                corr = stats.spearmanr(a, b)[0]
                print("Correlation " + cell + ": " + str(corr) + " [" + str(len(a)) + "]")
                corr = stats.spearmanr(ap, bp)[0]
                print("Correlation coding " + cell + ": " + str(corr) + " [" + str(len(ap)) + "]")

            del predictions
            # print("Drawing")
            # for c, cell in enumerate(cells):
            #     ci = -2 + c
            #     for i in range(1200, 1250, 1):
            #         fig, axs = plt.subplots(2, 1, figsize=(12, 8))
            #         vector1 = predictions[i][ci]
            #         vector2 = test_output[i][ci]
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
            #         plt.savefig(figures_folder + "/tracks/track_" + str(i + 1) + "_" + str(cell) + "_" + test_info[i] + ".png")
            #         plt.close(fig)
            #
            # # Marks
            # for m in range(10):
            #     for i in range(1200, 1250, 1):
            #         fig, axs = plt.subplots(2, 1, figsize=(12, 8))
            #         vector1 = predictions[i][m]
            #         vector2 = test_output[i][m]
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
            #         plt.savefig(figures_folder + "/marks/track_" + str(i + 1) + "_" + str(m) + "_" + test_info[i] + ".png")
            #         plt.close(fig)

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
        print("Cleaning" + datetime.now().strftime(' %H:%M:%S'))
        # Needed to prevent Keras memory leak
        del input_sequences
        del output_scores
        del our_model
        del gas
        gc.collect()
        K.clear_session()
        tf.compat.v1.reset_default_graph()
        print("Epoch " + str(k) + " finished. ")


# def read_ranges_1(file_path, good_chr):
#     counter = 0
#     ranges = {}
#     with open(file_path) as file:
#         for line in file:
#             if line.startswith("#"):
#                 continue
#             vals = line.split("\t")
#             info = vals[0].split("_")
#             chrn = info[0]
#             if chrn not in good_chr:
#                 continue
#             start = int(info[1]) - 1
#             end = int(info[2]) - 1
#             score = math.log(int(vals[4]))
#             ranges.setdefault(chrn, []).append([start, end, score])
#             counter = counter + 1
#     print(counter)
#     return ranges


def read_ranges_2(file_path, good_chr, pids, eids):
    counter = 0
    ranges_promoter = {}
    ranges_enhancer = {}
    with open(file_path) as file:
        for line in file:
            if line.startswith("#"):
                continue
            vals = line.split("\t")
            chrn = vals[0]
            if chrn not in good_chr:
                continue
            tss = int(vals[7]) - 1
            if vals[3] in pids:
                ranges_promoter.setdefault(chrn, []).append([tss, vals[3]])
            else:
                ranges_enhancer.setdefault(chrn, []).append([tss, vals[3]])
            counter = counter + 1
    print(counter)
    return ranges_promoter, ranges_enhancer


def get_seq(genome, chr, tss, input_size):
    half_size = math.floor(input_size / 2)
    seq = cm.encode_seq(genome[chr][tss - half_size:tss + half_size])
    return seq


if __name__ == '__main__':
    os.chdir(open("data_dir").read().strip())
    train()
