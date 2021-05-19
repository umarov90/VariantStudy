import math
import os
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

matplotlib.use("agg")


def train():
    input_size = 601
    num_regions = 4
    max_shift = 40
    good_chr = ["chrX", "chrY"]
    for i in range(2, 23):
        good_chr.append("chr" + str(i))

    if Path("genome.p").is_file():
        genome = pickle.load(open("genome.p", "rb"))
    else:
        genome = cm.parse_genome("hg19.fa")
        pickle.dump(genome, open("ranges.p", "wb"), protocol=pickle.HIGHEST_PROTOCOL)

    df = pd.read_csv("DMFB_IPSC.CRE.info.tsv", sep="\t")

    enhancers_ids = df.query("classc == 'distal'")['CREID'].to_list()
    promoters_ids = df.query("classc != 'distal'")['CREID'].to_list()

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

    num_cells = len(counts)
    model = m.simple_model(input_size, num_regions, num_cells)
    model.compile(loss="mse", optimizer=Adam(lr=1e-4))
    callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    if Path("input_sequences_long.p").is_file():
        input_sequences_long = pickle.load(open("input_sequences_long.p", "rb"))
        output_scores = pickle.load(open("output_scores.p", "rb"))
        test_input_sequences = pickle.load(open("test_input_sequences.p", "rb"))
        test_output = pickle.load(open("test_output.p", "rb"))
    else:
        test_input_sequences = []
        test_output = []
        for chr, chr_cres in test_promoters.items():
            for i in range(len(chr_cres)):
                cres = []
                scores = []
                chr_enhancers = test_enhancers[chr]
                enhs = nsmallest(num_regions - 1, chr_enhancers, key=lambda x: abs(x[0] - chr_cres[i][0]))
                for e in enhs:
                    if abs(e[0] - chr_cres[i][0]) < 200000:
                        cres.append(get_seq(genome, chr, e[0], input_size))
                        sub_scores = []
                        for cell in sorted(counts.keys()):
                            sub_scores.append(counts[cell][e[1]])
                        scores.append(sub_scores)
                while len(cres) < num_regions - 1:
                    cres.append(np.zeros((input_size, 4)).astype(np.bool))
                    scores.append(np.zeros(2))
                cres.append(get_seq(genome, chr, chr_cres[i][0], input_size))
                sub_scores = []
                for cell in sorted(counts.keys()):
                    sub_scores.append(counts[cell][chr_cres[i][1]])
                scores.append(sub_scores)
                test_input_sequences.append(cres)
                test_output.append(scores)
        test_input_sequences = np.asarray(test_input_sequences)
        test_output = np.asarray(test_output)

        input_sequences_long = []
        output_scores = []
        for chr, chr_cres in promoters.items():
            for i in range(len(chr_cres)):
                cres = []
                scores = []
                chr_enhancers = enhancers[chr]
                enhs = nsmallest(num_regions - 1, chr_enhancers, key=lambda x: abs(x[0] - chr_cres[i][0]))
                for e in enhs:
                    if abs(e[0] - chr_cres[i][0]) < 200000:
                        cres.append(get_seq(genome, chr, e[0], input_size + max_shift))
                        sub_scores = []
                        for cell in sorted(counts.keys()):
                            sub_scores.append(counts[cell][e[1]])
                        scores.append(sub_scores)
                while len(cres) < num_regions - 1:
                    cres.append(np.zeros((input_size + max_shift, 4)).astype(np.bool))
                    scores.append(np.zeros(2))
                cres.append(get_seq(genome, chr, chr_cres[i][0], input_size + max_shift))
                sub_scores = []
                for cell in sorted(counts.keys()):
                    sub_scores.append(counts[cell][chr_cres[i][1]])
                scores.append(sub_scores)
                input_sequences_long.append(cres)
                output_scores.append(scores)
        input_sequences_long = np.asarray(input_sequences_long)
        output_scores = np.asarray(output_scores)

        pickle.dump(input_sequences_long, open("input_sequences_long.p", "wb"), protocol=pickle.HIGHEST_PROTOCOL)
        pickle.dump(output_scores, open("output_scores.p", "wb"), protocol=pickle.HIGHEST_PROTOCOL)
        pickle.dump(test_input_sequences, open("test_input_sequences.p", "wb"), protocol=pickle.HIGHEST_PROTOCOL)
        pickle.dump(test_output, open("test_output.p", "wb"), protocol=pickle.HIGHEST_PROTOCOL)

    # input_sequences = []
    # for seq in test_input_sequences:
    #     new_arr = seq.reshape((seq.shape[0], -1))
    #     input_sequences.append(new_arr)
    # input_sequences = np.asarray(input_sequences)

    prev_model = None
    for k in range(150):
        input_sequences = []
        for seq in input_sequences_long:
            rand_var = 20 # random.randint(0, max_shift)
            ns = seq[:, rand_var: rand_var + input_size, :]
            # new_arr = ns.reshape((ns.shape[0], -1))
            input_sequences.append(ns)
        input_sequences = np.asarray(input_sequences)

        model.fit(input_sequences, output_scores, epochs=1, batch_size=64, validation_split=0.1,
                  callbacks=[callback])
        print("Epoch " + str(k))
        print("Training set")
        predictions = model.predict(input_sequences)

        for c, cell in enumerate(sorted(counts.keys())):
            a = []
            b = []
            for i in range(len(predictions)):
                a.append(predictions[i][-1][c])
                b.append(output_scores[i][-1][c])
            corr = stats.spearmanr(a, b)[0]
            print("Correlation " + cell + ": " + str(corr))
        print("Test set")
        predictions = model.predict(test_input_sequences)

        for c, cell in enumerate(sorted(counts.keys())):
            a = []
            b = []
            for i in range(len(predictions)):
                a.append(predictions[i][-1][c])
                b.append(test_output[i][-1][c])
            corr = stats.spearmanr(a, b)[0]
            print("Correlation " + cell + ": " + str(corr))

            a = []
            b = []
            for i in range(len(predictions)):
                if test_output[i][0][c] == 0:
                    continue
                a.append(predictions[i][0][c])
                b.append(test_output[i][0][c])
            corr = stats.spearmanr(a, b)[0]
            print("Enhancer 1 Correlation " + cell + ": " + str(corr))

            a = []
            b = []
            for i in range(len(predictions)):
                if test_output[i][1][c] == 0:
                    continue
                a.append(predictions[i][1][c])
                b.append(test_output[i][1][c])
            corr = stats.spearmanr(a, b)[0]
            print("Enhancer 2 Correlation " + cell + ": " + str(corr))
        # vector = model.predict(np.asarray([sequences[0]]))
        # plt.plot(vector)
        # plt.savefig("figures/track.png")


def read_ranges_1(file_path, good_chr):
    counter = 0
    ranges = {}
    with open(file_path) as file:
        for line in file:
            if line.startswith("#"):
                continue
            vals = line.split("\t")
            info = vals[0].split("_")
            chrn = info[0]
            if chrn not in good_chr:
                continue
            start = int(info[1]) - 1
            end = int(info[2]) - 1
            score = math.log(int(vals[4]))
            ranges.setdefault(chrn, []).append([start, end, score])
            counter = counter + 1
    print(counter)
    return ranges


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
    half_size = int((input_size - 1) / 2)
    seq = cm.encode_seq(genome[chr][tss - half_size:tss + half_size + 1])
    return seq


if __name__ == '__main__':
    os.chdir(open("data_dir").read().strip())
    train()
