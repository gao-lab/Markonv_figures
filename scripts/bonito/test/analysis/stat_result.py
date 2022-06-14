import sys
import os
import pandas as pd

result_dir = "../../../result/bonito/"

if len(sys.argv) == 2:
    filename = result_dir+sys.argv[1].replace(".fastq","_reads.tsv")
else:
    raise ValueError("Please input the filename.")

if os.path.exists(filename):
    result = pd.read_csv(filename,sep='\t')
    print('reads',result.median())

filename = filename.replace("reads","assembly")
if os.path.exists(filename):
    result = pd.read_csv(filename,sep='\t')
    print('assembly',result.median())

