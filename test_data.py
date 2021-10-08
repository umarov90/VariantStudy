import os
import joblib
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import gc

os.chdir(open("data_dir").read().strip())

avg_arr = []
directory = "parsed_data"

for i, filename in enumerate(os.listdir(directory)):
    if filename.endswith(".gz"):
        one_mat = joblib.load(os.path.join(directory, filename))
        avg_arr.append(one_mat)
    if len(avg_arr) > 1000:
        break


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