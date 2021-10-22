import pyBigWig
import numpy as np
import joblib
import os

os.chdir(open("data_dir").read().strip())

bin_size = 100
half_num_bins = 40
test_info = joblib.load("pickle/test_info.gz")
info = None

for v in test_info:
    if "ENSG00000162510.6" in v[2]:
        info = v
        break

bw = pyBigWig.open("/home/user/data/white_tracks/CAGE.RNA.ctss.Peripheral_Blood_Mononuclear_Cells_donor1.CNhs10860.FANTOM5.16nt.bigwig")
start = info[1] - half_num_bins * bin_size
end = info[1] + (1 + half_num_bins) * bin_size
out = bw.stats("chr1", start, end, type="mean", nBins=81, exact=True)
output = np.asarray(out)
bw.close()
output[output == None] = 0
output = output.astype(np.float32)
# output[np.isnan(output)] = 0
# output = np.log(output + 1)


bed = []

current_start = info[1] - half_num_bins * bin_size
for j, a in enumerate(output):
    start = current_start + j * bin_size
    line = f"{info[0]}\t{start}\t{start+bin_size}\t{a}"
    bed.append(line)

with open("cmb.bedgraph", 'w+') as f:
    f.write('\n'.join(bed))