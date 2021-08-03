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


def parse_tracks(ga, bin_size):
    gas_keys = []
    directory = "tracks"
    for filename in os.listdir(directory):
        if filename.endswith(".gz"):
            start = time.time()
            fn = os.path.join(directory, filename)
            t_name = fn.replace("/", "_")
            gas_keys.append(t_name)
            if Path("parsed_tracks/" + t_name).is_file():
                continue
            # print(t_name)

            gast = copy.deepcopy(ga)
            dtypes = {"chr": str, "start": int, "end": int, "score": float}
            df = pd.read_csv(fn, delim_whitespace=True, names=["chr", "start", "end", "score"],
                             dtype=dtypes, header=None, index_col=False)

            chrd = list(df["chr"].unique())
            df["mid"] = (df["start"] + (df["end"] - df["start"]) / 2) / bin_size
            df = df.astype({"mid": int})

            # group the scores over `key` and gather them in a list
            grouped_scores = df.groupby("chr").agg(list)

            # for each key, value in the dictionary...
            for key, val in gast.items():
                if key not in chrd:
                    continue
                # first lookup the positions to update and the corresponding scores
                pos, score = grouped_scores.loc[key, ["mid", "score"]]
                # fancy indexing
                gast[key][pos] += score

            max_val = -1
            for key in gast.keys():
                gast[key] = np.log(gast[key] + 1)
                max_val = max(np.max(gast[key]), max_val)
            for key in gast.keys():
                gast[key] = gast[key] / max_val
            for key in gast.keys():
                gast[key] = gast[key].astype(np.float16)
            joblib.dump(gast, "parsed_tracks/" + t_name, compress="lz4")
            end = time.time()
            print("Parsed " + t_name + ". Elapsed time: " + str(end - start) + ". Max value: " + str(max_val))
    joblib.dump(gas_keys, "pickle/gas_keys.gz", compress=3)
    return gas_keys


def get_sequences(bin_size, chromosomes):
    if Path("pickle/genome.gz").is_file():
        genome = joblib.load("pickle/genome.gz")
        ga = joblib.load("pickle/ga.gz")
    else:
        genome, ga = cm.parse_genome("hg38.fa", bin_size)
        joblib.dump(genome, "pickle/genome.gz", compress=3)
        joblib.dump(ga, "pickle/ga.gz", compress=3)

    if Path("pickle/one_hot.gz").is_file():
        one_hot = joblib.load("pickle/one_hot.gz")
    else:
        one_hot = {}
        for chromosome in chromosomes:
            one_hot[chromosome] = cm.encode_seq(genome[chromosome])
        joblib.dump(one_hot, "pickle/one_hot.gz", compress=3)

    if Path("pickle/train_info.gz").is_file():
        test_info = joblib.load("pickle/test_info.gz")
        train_info = joblib.load("pickle/train_info.gz")
    else:
        gene_tss = pd.read_csv("TSS_flank_0.bed",
                            sep="\t", index_col=False, names=["chrom", "start", "end", "geneID", "score", "strand"])
        gene_info = pd.read_csv("gene.info.tsv",
                               sep="\t", index_col=False)
        test_info = []
        test_genes = gene_tss.loc[gene_tss['chrom'] == "chr1"]
        for index, row in test_genes.iterrows():
            pos = int(row["start"])
            gene_type = gene_info[gene_info['geneID'] == row["geneID"]]['geneType'].values[0]
            test_info.append([row["chrom"], pos, row["geneID"], gene_type, row["strand"]])
        print(f"Test set complete {len(test_info)}")
        train_info = []
        train_genes = gene_tss.loc[gene_tss['chrom'] != "chr1"]
        for index, row in train_genes.iterrows():
            pos = int(row["start"])
            gene_type = gene_info[gene_info['geneID'] == row["geneID"]]['geneType'].values[0]
            train_info.append([row["chrom"], pos, row["geneID"], gene_type, row["strand"]])
        print(f"Training set complete {len(train_info)}")
        joblib.dump(test_info, "pickle/test_info.gz", compress=3)
        joblib.dump(train_info, "pickle/train_info.gz", compress=3)
        gc.collect()
    return ga, one_hot, train_info, test_info


def parse_one_track(ga, bin_size, fn):
    gast = copy.deepcopy(ga)
    dtypes = {"chr": str, "start": int, "end": int, "score": float}
    df = pd.read_csv(fn, delim_whitespace=True, names=["chr", "start", "end", "score"],
                     dtype=dtypes, header=None, index_col=False)

    chrd = list(df["chr"].unique())
    df["mid"] = (df["start"] + (df["end"] - df["start"]) / 2) / bin_size
    df = df.astype({"mid": int})

    # group the scores over `key` and gather them in a list
    grouped_scores = df.groupby("chr").agg(list)

    # for each key, value in the dictionary...
    for key, val in gast.items():
        if key not in chrd:
            continue
        # first lookup the positions to update and the corresponding scores
        pos, score = grouped_scores.loc[key, ["mid", "score"]]
        # fancy indexing
        gast[key][pos] += score

    max_val = -1
    for key in gast.keys():
        gast[key] = np.log(gast[key] + 1)
        max_val = max(np.max(gast[key]), max_val)
    for key in gast.keys():
        gast[key] = gast[key] / max_val

    return gast