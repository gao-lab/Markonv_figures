"""The script is used to generate the read_id_to_fast5.tab, which is necessary for fix_read_names.py.
"""

import pdb


with open("read_id_to_fast5.tab", "w") as g:
    with open("tail_dna_r9.4.1@v3.3.fastq","r") as f:
        for i, line in enumerate(f.read().split('\n')):
            if i%4 == 0:
                line = line.split('\t')
                if len(line)>1:
                    g.write(line[0][1:].split(' ')[0]+'\t'+line[8]+'\n')
