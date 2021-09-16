import os
import pandas as pd

os.chdir(open("data_dir").read().strip())
meta = pd.read_csv("data/parsed.meta_data.tsv", sep="\t", index_col="tag")
print(len(meta.index))
meta["file_size"] = 1
directory = "bw"
for filename in os.listdir(directory):
    if filename.endswith(".bigwig"):
        tag = filename[:-len(".16nt.bigwig")]
        fn = os.path.join(directory, filename)
        size = os.path.getsize(fn)
        meta.at[tag, "file_size"] = size

s_maxes = meta.groupby(["Assay", "Experiment_target", "Biosample_term_name"]).file_size.transform(max)
meta = meta.loc[meta.file_size == s_maxes]
print(len(meta.index))

with open('white_list.txt', 'w') as f:
    for item in meta.index:
        f.write("%s\n" % item)
