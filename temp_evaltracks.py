import os
import joblib

os.chdir(open("data_dir").read().strip())

eval_track_names = joblib.load("eval_track_names.gz")
for tr in eval_track_names:
    print(tr)
