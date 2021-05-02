import os
import model as m
import numpy as np
import tensorflow as tf
import common as cm
import pickle
from tensorflow.keras.optimizers import Adam
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use("agg")


def train():
    input_size = 10001
    good_chr = ["chrX", "chrY"]
    for i in range(2, 23):
        good_chr.append("chr" + str(i))
    model = m.simple_model(input_size)
    # genes = get_genes(good_chr)
    # pickle.dump(genes, open("variants/genes.p", "wb"), protocol=pickle.HIGHEST_PROTOCOL)
    genes = pickle.load(open("variants/genes.p", "rb"))
    # genome = cm.parse_genome("data/genomes/hg19.fa")
    # pickle.dump(genome, open("variants/genome.p", "wb"), protocol=pickle.HIGHEST_PROTOCOL)
    genome = pickle.load(open("variants/genome.p", "rb"))
    # ga = {}
    # for key in genome.keys():
    #     ga[key] = np.zeros((len(genome[key])), dtype=np.uint8)
    # pickle.dump(ga, open("variants/ga.p", "wb"), protocol=pickle.HIGHEST_PROTOCOL)
    ga = pickle.load(open("variants/ga.p", "rb"))
    sequences = get_sequences(genome, genes, input_size)
    expression_tracks = sample_tracks(good_chr, genes, input_size, ga.copy())
    model.compile(loss="mse", optimizer=Adam(lr=1e-5))
    callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    for i in range(len(sequences) - 1, -1, -1):
        if len(sequences[i]) != input_size:
            del sequences[i]
            del expression_tracks[i]

    sequences = np.asarray(sequences)
    expression_tracks = np.asarray(expression_tracks)
    model.fit(sequences, expression_tracks, epochs=100, batch_size=32, validation_split=0.1,
              callbacks=[callback])
    vector = model.predict(np.asarray([sequences[0]]))
    plt.plot(vector)
    plt.savefig("figures_v/track.png")


def get_genes(good_chr):
    counter = 0
    genes = {}
    with open("data/gencode.v34lift37.annotation.gff3") as file:
        for line in file:
            if line.startswith("#"):
                continue
            vals = line.split("\t")
            chrn = vals[0]
            if chrn not in good_chr:
                continue
            if vals[2] != "gene":
                continue
            start = int(vals[3]) - 1
            genes.setdefault(chrn, []).append(start)
            counter = counter + 1
    print(counter)
    return genes


def get_sequences(genome, genes, input_size):
    sequences = []
    half_size = int(input_size / 2)
    for key in genes.keys():
        for gene in genes[key]:
            sequences.append(cm.encode_seq(genome[key][gene-half_size:gene+half_size + 1]))
    return sequences


def sample_tracks(good_chr, genes, input_size, ga):
    tracks = []
    half_size = int(input_size / 2)

    with open('data/hg19.cage_peak_phase1and2combined_coord.bed') as file:
        for line in file:
            vals = line.split("\t")
            chrn = vals[0]
            score = int(vals[4])
            strand = vals[5]
            if chrn not in good_chr:
                continue
            chrp = int(vals[7]) - 1
            ga[chrn][chrp] = score

    for key in genes.keys():
        for gene in genes[key]:
            tracks.append(ga[key][gene - half_size:gene + half_size + 1])
    return tracks


if __name__ == '__main__':
    os.chdir(open("data_dir").read().strip())
    train()
