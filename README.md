# Markonv: a novel convolutional layer for identifying inter-positional correlations effectively and efficiently
This is the repository for reproducing figures and tables in the paper.

## Requirements

### Python environment

```bash
conda env create -f env_markonv.yaml
conda activate markonv
pip3 install git+https://github.com/rrwick/Porechop.git
cd scripts/bonito
python setup.py develop
```

### R packages

- ggpubr
- data.table
- magrittr
- foreach

## Reproduce results related to simulations
1. Generate simulation datasets and plot Markov transition matrix (Reproduce Appendix C)

```bash
cd external/simulation
python3 GeneRateMarcov.py
```
The plotted Markov transition matrix is saved in `external/simulation/motif/`.

2. Training and evaluation

```bash
cd ../../scripts/simulation
python3 torch_trainMarkonv.py
```

3. Plot AUROC (Reproduce Figure 2)

```bash
python3 Compare.py
Rscript generate_fig_2.R
```
The figure generated is saved in `result/simulation.auroc.png`.

4. Recover motifs (Reproduce Figure 3 and Appendix G)

```bash
python3 geneRatemotifs.py
python3 kernel2motif.py
cd ../../
```
The recovered Markov transition matrix is saved in `result/simulation/Motifs/*.png`.

You can then compare the recovered motif with the real motif. Offset may exist between the real motif and the recovered motif, and the offset is saved in `result/simulation/Motifs/kernel_offset.txt`.

## Reproduce results related to HOCNNLB (Figure 4)

Downloads training and test datasets from https://github.com/NWPU-903PR/HOCNNLB/blob/master/lncRBPdata.zip and put them into ./external/HOCNNLB

```bash
cd external/HOCNNLB 
unzip lncRBPdata.zip
mv RBPdata1201 fasta
cd ../code/
python generateHDF5.py

cd ../../../scripts/HOCNNLB
python torch_trainMarkonv.py

python Compare.py

cd ../../
```

## Reproduce results related to Bonito (Table 2)
1. Download training and test sets

Get training set
```bash
cd scripts/bonito
bonito download --training
```

Get test set
```bash
mkdir -p ../../external/bonito/Klebsiella_pneumoniae_INF032_fast5s
cd ../../external/bonito/Klebsiella_pneumoniae_INF032_fast5s
wget https://bridges.monash.edu/ndownloader/files/15188573
mv 15188573 Klebsiella_pneumoniae_INF032_fast5s.tar.gz
tar -xzf Klebsiella_pneumoniae_INF032_fast5s.tar.gz

cd ../../../scripts/bonito/test
wget https://bridges.monash.edu/ndownloader/files/14260223
mv 14260223 Klebsiella_pneumoniae_INF032_reference.fasta.gz
gzip -d Klebsiella_pneumoniae_INF032_reference.fasta.gz
mv Klebsiella_pneumoniae_INF032_reference.fasta reference.fasta
cd ../
```

2. Training and basecalling

Convolution-based bonito network
```
bash run_train_basecall_conv.sh
```

Markonv-based bonito network
```
bash run_train_basecall_markonv.sh
```

The number of parameters for each network (in Table 2) will be printed on the screen.

3. Evaluating basecalled reads

Convolution-based bonito network
```
cd test
bash run_analysis_conv.sh
```

Markonv-based bonito network
```
bash run_analysis_markonv.sh
cd ../../../
```
The read accuracy for each read is saved in `result/bonito/*_reads.tsv`, and the consensus accuracy for each read is saved in `result/bonito/*_assembly.tsv`.
The median of read accuracy and consensus accuracy for each network (in Table 2) will be printed on the screen.


## Reproduce results related to loss curve (Figure 5 and Appdenix H)
```
cd scripts/SpeedCompare
python torch_trainMarkonv.py
python DrawTheHistory.py
```
The figure is saved in `result/SpeedCompare/figure/`.
