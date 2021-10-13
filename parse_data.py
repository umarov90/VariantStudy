import os
from pathlib import Path
import pandas as pd
import numpy as np
import joblib
import gc
import common as cm
import re
import math
import time
import copy
import itertools as it
import pyBigWig
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import random
from multiprocessing import Pool, Manager
import multiprocessing as mp


def valid(chunks):
    for chunk in chunks:
        print("1")
        mask = chunk['chr1'] == chunk['chr2']
        if mask.all():
            yield chunk
        else:
            yield chunk.loc[mask]
            break


def parse_hic():
    if Path("pickle/hic_keys.gz").is_file():
        return joblib.load("pickle/hic_keys.gz")
    else:
        hic_keys = []
        directory = "hic"

        for filename in os.listdir(directory):
            if filename.endswith(".bz2"):
                fn = os.path.join(directory, filename)
                t_name = fn.replace("/", "_")
                print(t_name)
                hic_keys.append(t_name)
                if Path("parsed_hic/" + t_name + "chr1").is_file():
                    continue
                with open("hic.txt", "a+") as myfile:
                    myfile.write(t_name)
                fields = ["chr1", "chr2", "locus1", "locus2", "pvalue"]
                dtypes = {"chr1": str, "chr2": str, "locus1": int, "locus2": int, "pvalue": str}
                chunksize = 10 ** 8
                chunks = pd.read_csv(fn, sep="\t", index_col=False, usecols=fields,
                                     dtype=dtypes, chunksize=chunksize, low_memory=True)
                df = pd.concat(valid(chunks))
                # df = pd.read_csv(fn, sep="\t", index_col=False, usecols=fields, dtype=dtypes, low_memory=True)
                df['pvalue'] = pd.to_numeric(df['pvalue'], errors='coerce')
                df['pvalue'].fillna(0, inplace=True)
                print(len(df))
                df.drop(df[df['chr1'] != df['chr2']].index, inplace=True)
                print(len(df))
                df.drop(['chr2'], axis=1, inplace=True)
                df.drop(df[df['locus1'] - df['locus2'] > 1000000].index, inplace=True)
                print(len(df))
                df["pvalue"] = -1 * np.log(df["pvalue"])
                m = df.loc[df['pvalue'] != np.inf, 'pvalue'].max()
                print("P Max is: " + str(m))
                df['pvalue'].replace(np.inf, m, inplace=True)
                df['pvalue'].clip(upper=100, inplace=True)
                df["score"] = df["pvalue"] / df["pvalue"].max()
                df.drop(["pvalue"], axis=1, inplace=True)
                chrd = list(df["chr1"].unique())
                for chr in chrd:
                    joblib.dump(df.loc[df['chr1'] == chr].sort_values(by=['locus1']),
                                "parsed_hic/" + t_name + chr, compress=3)
                print(t_name)
                with open("hic.txt", "a+") as myfile:
                    myfile.write(t_name)
                del df
                gc.collect()

        joblib.dump(hic_keys, "pickle/hic_keys.gz", compress=3)
        chromosomes = ["chrX", "chrY"]
        for i in range(1, 23):
            chromosomes.append("chr" + str(i))
        for key in hic_keys:
            print(key)
            hdf = {}
            for chr in chromosomes:
                hdf[chr] = joblib.load("parsed_hic/" + key + chr)
            joblib.dump(hdf, "parsed_hic/" + key, compress=3)
            print(key)
        return hic_keys


def parse_tracks(train_info, test_info, bin_size, half_num_bins):
    all_info = train_info + test_info
    track_names = pd.read_csv('data/white_list.txt', delimiter='\t').values.flatten().tolist()
    step_size = 100
    q = mp.Queue()
    ps = []
    start = 0
    end = len(all_info)
    for t in range(start, end, step_size):
        t_end = min(t+step_size, end)
        sub_info = all_info[t:t_end]
        p = mp.Process(target=construct_tss_matrices,
                       args=(q, sub_info, half_num_bins, bin_size,track_names,t,))
        p.start()
        ps.append(p)
        if len(ps) >= 12:
            for p in ps:
                p.join()
            print(q.get())
            ps = []
        print(f"reached {t}")

    if len(ps) > 0:
        for p in ps:
            p.join()
        print(q.get())

    directory = "parsed_data"
    all_maxes = None
    for i, filename in enumerate(os.listdir(directory)):
        if i % 500 == 0:
            print(i, end=" ")
            all_maxes = np.max(all_maxes, axis=0)
            gc.collect()
        if filename.endswith(".gz"):
            one_mat = joblib.load(os.path.join(directory, filename))
            maxes = one_mat.max(axis=1)
            if all_maxes is None:
                all_maxes = maxes
            else:
                all_maxes = np.row_stack((all_maxes, maxes))

    all_maxes = np.max(all_maxes, axis=0)
    print(f"max {np.max(all_maxes)} min {np.min(all_maxes)} mean {np.mean(all_maxes)} median {np.median(all_maxes)}")
    np.savetxt("all_maxes.csv", all_maxes, delimiter=",")

    for i, filename in enumerate(os.listdir(directory)):
        if i % 500 == 0:
            print(i, end=" ")
            gc.collect()
        if filename.endswith(".gz"):
            one_mat = joblib.load(os.path.join(directory, filename))
            one_mat = one_mat / all_maxes[:, None]
            if not np.isfinite(one_mat).all():
                print("Problem!" + filename)
            joblib.dump(one_mat, "parsed_data_processed/" + filename, compress="lz4")

    joblib.dump(track_names, "pickle/track_names.gz", compress=3)
    return track_names


def construct_tss_matrices(q, sub_info, half_num_bins, bin_size, track_names, t):
    print(f"Worker {t} - got {len(sub_info)} TSS to process")
    output = np.zeros((len(sub_info), len(track_names), half_num_bins*2 + 1))
    for ti, track in enumerate(track_names):
        if ti % 1000 == 0:
            print(f"Worker {t} - {ti}")
        bw = pyBigWig.open(f"/home/user/bw/{track}.16nt.bigwig")
        for i, info in enumerate(sub_info):
            start = info[1] - half_num_bins * bin_size
            end = info[1] + (1 + half_num_bins) * bin_size
            out = bw.stats(info[0], start, end, type="mean", nBins=half_num_bins*2 + 1)
            output[i, track_names.index(track), :] = out
        bw.close()
    output = np.log10(output + 1)
    output[np.isnan(output)] = 0
    output = output.astype('float32')
    # print(np.asarray(output).shape)
    for i in range(len(sub_info)):
        joblib.dump(np.asarray(output[i]), "parsed_data/" + str(sub_info[i][-1]) + ".gz", compress="lz4")
    q.put(None)


def get_sequences(chromosomes, input_size):
    if Path("pickle/one_hot.gz").is_file():
        one_hot = joblib.load("pickle/one_hot.gz")
    else:
        print("Parsing genome")
        genome = cm.parse_genome("data/hg38.fa", chromosomes)
        one_hot = {}
        for chromosome in chromosomes:
            one_hot[chromosome] = cm.encode_seq(genome[chromosome])
        print("Saving one-hot encoding")
        joblib.dump(one_hot, "pickle/one_hot.gz", compress=3)

    if Path("pickle/train_info.gz").is_file():
        test_info = joblib.load("pickle/test_info.gz")
        train_info = joblib.load("pickle/train_info.gz")
    else:
        gene_tss = pd.read_csv("data/hg38.GENCODEv38.pc_lnc.TSS.bed",
                               sep="\t", index_col=False, names=["chrom", "start", "end", "geneID", "score", "strand"])
        gene_info = pd.read_csv("data/hg38.GENCODEv38.pc_lnc.gene.info.tsv", sep="\t", index_col=False)
        prom_info = pd.read_csv("data/hg38.GENCODEv38.pc_lnc.promoter.window.info.tsv", sep="\t", index_col=False)
        test_info = []
        # test_genes = prom_info.loc[(prom_info['chrom'] == "chr1") & (prom_info['max_overall_rank'] == 1)]
        # for index, row in test_genes.iterrows():
        #     vals = row["TSS_str"].split(";")
        #     pos = int(vals[int(len(vals) / 2)].split(",")[1])
        #     strand = vals[int(len(vals) / 2)].split(",")[2]
        #     test_info.append([row["chrom"], pos, row["geneID_str"], row["geneType_str"], strand])
        print("Constructing test and train genes list")
        test_genes = gene_tss.loc[gene_tss['chrom'] == "chr1"]
        for index, row in test_genes.iterrows():
            pos = int(row["end"])
            if row["chrom"] not in chromosomes or pos - input_size / 2 < 0 or pos + input_size > len(one_hot[row["chrom"]]):
                continue
            gene_type = gene_info[gene_info['geneID'] == row["geneID"]]['geneType'].values[0]
            if gene_type != "protein_coding":
                continue
            test_info.append([row["chrom"], pos, row["geneID"], gene_type, row["strand"], row["geneID"] + "_" + str(pos)])

        print(f"Test set complete {len(test_info)}")
        train_info = []
        train_genes = gene_tss.loc[gene_tss['chrom'] != "chr1"]
        for index, row in train_genes.iterrows():
            pos = int(row["end"])
            if row["chrom"] not in chromosomes or pos - input_size / 2 < 0 or pos + input_size > len(one_hot[row["chrom"]]):
                continue
            gene_type = gene_info[gene_info['geneID'] == row["geneID"]]['geneType'].values[0]
            if gene_type != "protein_coding":
                continue
            train_info.append(
                [row["chrom"], pos, row["geneID"], gene_type, row["strand"], row["geneID"] + "_" + str(pos)])
        print(f"Training set complete {len(train_info)}")
        joblib.dump(test_info, "pickle/test_info.gz", compress=3)
        joblib.dump(train_info, "pickle/train_info.gz", compress=3)
        gc.collect()
    return one_hot, train_info, test_info
