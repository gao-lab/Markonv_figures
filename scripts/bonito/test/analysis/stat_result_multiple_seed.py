import sys
import os
import pandas as pd

result_dir = "../../../result/bonito/"

for prefix in ["dna_r9.4.1@v3.3","mar38v_hac"]:
    all_read_acc = []
    all_ass_acc = []
    for seed in [None, 42, 23, 123, 345, 1234, 9, 2323, 927, 48, 100, 420, 60, 320, 7767, 51]:
        if seed is None:
            filename = result_dir+f"fix_{prefix}_reads.tsv"
        else:
            filename = result_dir+f"fix_{prefix}_{seed}_reads.tsv"

        if os.path.exists(filename):
            result = pd.read_csv(filename,sep='\t')
            read_acc = result["Identity"].median()
            all_read_acc.append(read_acc)

        filename = filename.replace("reads","assembly")
        if os.path.exists(filename):
            result = pd.read_csv(filename,sep='\t')
            ass_acc = result["Identity"].median()
            all_ass_acc.append(ass_acc)

    with open(f"{result_dir}read_{prefix}.txt", "w") as f:
        f.write('\n'.join(list(map(str, all_read_acc))))
    with open(f"{result_dir}ass_{prefix}.txt", "w") as f:
        f.write('\n'.join(list(map(str, all_ass_acc))))
        