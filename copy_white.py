import os
import pandas as pd
from shutil import copy2


os.chdir(open("data_dir").read().strip())
track_names = pd.read_csv('white_list.txt', delimiter='\t').values.flatten()
for i, track in enumerate(track_names):
    if i % 100 == 0:
        print(i, end=" ")
    fn = f"bw/{track}.16nt.bigwig"
    copy2(fn, "/home/user/data/white_tracks")

