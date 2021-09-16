import os

os.chdir(open("data_dir").read().strip())
white_list = []
directory = "bw"
for filename in os.listdir("bw"):
    if filename.endswith(".bigwig"):
        tag = filename[:-len(".16nt.bigwig")]
        if "CAGE.RNA.ctss" in tag:
            fn = os.path.join(directory, filename)
            size = os.path.getsize(fn)
            if size > 512000:
                white_list.append(tag)

with open('white_list.txt', 'a+') as f:
    for item in white_list:
        f.write("%s\n" % item)
