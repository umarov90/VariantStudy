import os
import joblib
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

os.chdir(open("data_dir").read().strip())

avg_arr = []
directory = "parsed_data"
all_maxes = None
for filename in os.listdir(directory):
    if filename.endswith(".gz"):
        one_mat = joblib.load(os.path.join(directory, filename))
        one_mat[one_mat == np.inf] = np.finfo(np.float16).max
        maxes = one_mat.max(axis=1)
        if all_maxes is None:
            all_maxes = maxes
        else:
            all_maxes = np.row_stack((all_maxes, maxes))

all_maxes = np.max(all_maxes, axis=0)
all_maxes = np.log(all_maxes + 1)
print(f"max {np.max(all_maxes)} min {np.min(all_maxes)} mean {np.mean(all_maxes)} median {np.median(all_maxes)}")
np.savetxt("all_maxes.csv", all_maxes, delimiter=",")

for filename in os.listdir(directory):
    if filename.endswith(".gz"):
        one_mat = joblib.load(os.path.join(directory, filename))
        one_mat = np.log(one_mat + 1)
        one_mat = one_mat / all_maxes[:, None]
        joblib.dump(one_mat, "parsed_data_processed/" + filename, compress="lz4")
        avg_arr.append(one_mat)

track_names = pd.read_csv('data/white_list.txt', delimiter='\t').values.flatten()
avg_arr = np.mean(np.asarray(avg_arr), axis=0)
for i in range(len(avg_arr)):
    sub_arr = avg_arr[i]
    x = range(len(sub_arr))
    d2 = {'bin': x, 'expression': sub_arr}
    df = pd.DataFrame(d2)
    fig, axs = plt.subplots(1, 1, figsize=(8, 5))
    sns.lineplot(data=df, x='bin', y='expression')
    fig.tight_layout()
    plt.savefig(f"test_figs/{track_names[i]}.png")
    plt.close(fig)