import math
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = '2'
import random
import pandas as pd
import attention
import model
import model as m
import numpy as np
import tensorflow as tf
import common as cm
from pathlib import Path
import pickle
from tensorflow.keras.optimizers import Adam
from scipy import stats
import matplotlib
import matplotlib.pyplot as plt
from heapq import nsmallest
from tensorflow import keras
import copy
import seaborn as sns
matplotlib.use("agg")
# tf.compat.v1.disable_eager_execution()

def train():
    # implement cropping. only half of the input for binning
    # Apply smoothing to output bins? Ask Hon how to do it best
    # transformer_layers = 8 Try 1 instead of 8
    # remove last mlp to get 0.6
    input_size = 100400
    half_size = input_size / 2
    max_shift = 10
    bin_size = 400
    num_regions = int(input_size / bin_size)
    mid_bin = math.floor(num_regions / 2)
    good_chr = ["chrX", "chrY"]
    # our_model = m.simple_model(input_size, num_regions, 2)
    for i in range(2, 23):
        good_chr.append("chr" + str(i))

    if Path("genome.p").is_file():
        genome = pickle.load(open("genome.p", "rb"))
        ga = pickle.load(open("ga.p", "rb"))
    else:
        genome, ga = cm.parse_genome("hg19.fa", bin_size)
        pickle.dump(genome, open("genome.p", "wb"), protocol=pickle.HIGHEST_PROTOCOL)
        pickle.dump(ga, open("ga.p", "wb"), protocol=pickle.HIGHEST_PROTOCOL)

    df = pd.read_csv("DMFB_IPSC.CRE.info.tsv", sep="\t")

    enhancers_ids = df.query("classc == 'distal'")['CREID'].to_list()
    promoters_ids = df.query("classc != 'distal'")['CREID'].to_list()
    coding_ids = df.query("classc == 'coding'")['CREID'].to_list()

    ranges_file = "DMFB_IPSC.CRE.coord.bed"
    promoters, enhancers = read_ranges_2(ranges_file, good_chr, promoters_ids, enhancers_ids)
    test_promoters, test_enhancers = read_ranges_2(ranges_file, ["chr1"], promoters_ids, enhancers_ids)


    if Path("counts.p").is_file():
        counts = pickle.load(open("counts.p", "rb"))
    else:
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
        pickle.dump(counts, open("counts.p", "wb"), protocol=pickle.HIGHEST_PROTOCOL)

    num_cells = len(counts)
    cells = list(sorted(counts.keys()))

    if Path("input_sequences_long.p").is_file():
        input_sequences_long = pickle.load(open("input_sequences_long.p", "rb"))
        output_scores = pickle.load(open("output_scores.p", "rb"))
        test_input_sequences = pickle.load(open("test_input_sequences.p", "rb"))
        test_output = pickle.load(open("test_output.p", "rb"))
        test_class = pickle.load(open("test_class.p", "rb"))
    else:
        gas = {}
        for cell in cells:
            gas[cell] = copy.deepcopy(ga)

        for chr in good_chr:
            for p in promoters[chr] + enhancers[chr]:
                for cell in cells:
                    pos = int(p[0] / bin_size)
                    gas[cell][chr][pos] += counts[cell][p[1]]
                    if pos - 1 > 0:
                        gas[cell][chr][pos - 1] += counts[cell][p[1]]
                    if pos + 1 < num_regions:
                        gas[cell][chr][pos + 1] += counts[cell][p[1]]

        for chr in ["chr1"]:
            for p in test_promoters[chr] + test_enhancers[chr]:
                for cell in cells:
                    pos = int(p[0] / bin_size)
                    gas[cell][chr][pos] += counts[cell][p[1]]
                    if pos - 1 > 0:
                        gas[cell][chr][pos - 1] += counts[cell][p[1]]
                    if pos + 1 < num_regions:
                        gas[cell][chr][pos + 1] += counts[cell][p[1]]
        test_input_sequences = []
        test_output = []
        test_class = []
        for chr, chr_cres in test_promoters.items():
            for i in range(len(chr_cres)):
                tss = chr_cres[i][0]
                seq = get_seq(genome, chr, tss, input_size)
                if len(seq) != input_size:
                    continue
                test_input_sequences.append(seq)
                start = int((tss - half_size) / bin_size)
                scores = []
                for cell in cells:
                    scores.append(gas[cell][chr][start: start + num_regions])
                test_output.append(scores)
                if chr_cres[i][1] in coding_ids:
                    test_class.append(1)
                else:
                    test_class.append(0)

        test_input_sequences = np.asarray(test_input_sequences)
        test_output = np.asarray(test_output)

        input_sequences_long = []
        output_scores = []
        for chr, chr_cres in promoters.items():
            for i in range(len(chr_cres)):
                tss = chr_cres[i][0]
                seq = get_seq(genome, chr, tss, input_size + max_shift)
                if len(seq) != input_size + max_shift:
                    continue
                input_sequences_long.append(seq)
                start = int((tss - half_size) / bin_size)
                scores = []
                for cell in cells:
                    scores.append(gas[cell][chr][start: start + num_regions])
                output_scores.append(scores)
        input_sequences_long = np.asarray(input_sequences_long)
        output_scores = np.asarray(output_scores)

        pickle.dump(input_sequences_long, open("input_sequences_long.p", "wb"), protocol=pickle.HIGHEST_PROTOCOL)
        pickle.dump(output_scores, open("output_scores.p", "wb"), protocol=pickle.HIGHEST_PROTOCOL)
        pickle.dump(test_input_sequences, open("test_input_sequences.p", "wb"), protocol=pickle.HIGHEST_PROTOCOL)
        pickle.dump(test_output, open("test_output.p", "wb"), protocol=pickle.HIGHEST_PROTOCOL)
        pickle.dump(test_class, open("test_class.p", "wb"), protocol=pickle.HIGHEST_PROTOCOL)

    model_folder = "model1"
    model_name = "resnet.h5"

    if os.path.exists(model_folder) and os.path.isdir(model_folder) and os.listdir(model_folder):
        # our_model.load_weights(model_folder + "/" + model_name)
        print("")
    else:
        Path(model_folder).mkdir(parents=True, exist_ok=True)

    # strategy = tf.distribute.MirroredStrategy()
    # print('Number of devices: {}'.format(strategy.num_replicas_in_sync))
    # with strategy.scope():
    our_model = m.simple_model(input_size, num_regions, num_cells)
    our_model.compile(loss="mse", optimizer=Adam(lr=0.0001))
    BATCH_SIZE = 8
    for k in range(150):
        input_sequences = []
        for seq in input_sequences_long:
            rand_var = random.randint(0, max_shift)
            # rand_var = 5
            ns = seq[rand_var: rand_var + input_size, :]
            # new_arr = ns.reshape((ns.shape[0], -1))
            input_sequences.append(ns)
        input_sequences = np.asarray(input_sequences)

        our_model.fit(input_sequences, output_scores, epochs=1, batch_size=BATCH_SIZE)
        our_model.save_weights(model_folder + "/" + model_name)
        print("Epoch " + str(k))
        print("Training set")
        predictions = our_model.predict(input_sequences[0:3000], batch_size=BATCH_SIZE)
        # , batch_size=1, workers=1, use_multiprocessing=False, max_queue_size=1

        for c, cell in enumerate(cells):
            a = []
            b = []
            for i in range(len(predictions)):
                # if output_scores[i][c][mid_bin] == 0:
                #     continue
                a.append(predictions[i][c][mid_bin])
                b.append(output_scores[i][c][mid_bin])
            corr = stats.spearmanr(a, b)[0]
            print("Correlation " + cell + ": " + str(corr))
        print("Test set")
        predictions = our_model.predict(test_input_sequences, batch_size=BATCH_SIZE)

        for c, cell in enumerate(cells):
            a = []
            b = []
            ap = []
            bp = []
            for i in range(len(predictions)):
                # if test_output[i][c][mid_bin] == 0:
                #     continue
                a.append(predictions[i][c][mid_bin])
                b.append(test_output[i][c][mid_bin])
                if test_class[i] == 0:
                    continue
                ap.append(predictions[i][c][mid_bin])
                bp.append(test_output[i][c][mid_bin])
            corr = stats.pearsonr(a, b)[0]
            print("Correlation " + cell + ": " + str(corr) + " [" + str(len(a)) + "]")
            corr = stats.pearsonr(ap, bp)[0]
            print("Correlation coding " + cell + ": " + str(corr) + " [" + str(len(ap)) + "]")

        for i in range(200, 300, 1):
            fig, axs = plt.subplots(2, 1, figsize=(12, 8))
            vector1 = predictions[i][1]
            vector2 = test_output[i][1]
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
            plt.savefig("figures/track_" + str(i + 1) + ".png")
            plt.close(fig)

        # Gene regplot

        a = []
        b = []
        for i in range(len(predictions)):
            if test_class[i] == 0:
                continue
            a.append(predictions[i][1][mid_bin])
            b.append(test_output[i][1][mid_bin])

        pickle.dump(a, open("a.p", "wb"), protocol=pickle.HIGHEST_PROTOCOL)
        pickle.dump(b, open("b.p", "wb"), protocol=pickle.HIGHEST_PROTOCOL)

        fig, ax = plt.subplots(figsize=(6, 6))
        r, p = stats.pearsonr(a, b)

        sns.regplot(x=a, y=b,
                    ci=None, label="r = {0:.2f}; p = {1:.2e}".format(r, p)).legend(loc="best")

        ax.set(xlabel='Predicted', ylabel='Ground truth')
        plt.title("Gene expression prediction")
        fig.tight_layout()
        plt.savefig("figures/corr" + str(k) + ".svg")


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
