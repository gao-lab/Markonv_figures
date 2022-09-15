#!/bin/bash
#SBATCH -J conv_bonito
#SBATCH -N 1
#SBATCH -o conv_%j.out
#SBATCH -e conv_%j.err
#SBATCH --gres=gpu:1
#SBATCH -c 8

model_name=dna_r9.4.1@v3.3
seed=$1
model_dir=../../models/"$model_name"_"$seed"
test_dir=../bonito/test/
result_dir=../../result/bonito/

# 1. View model
bonito view ../bonito/bonito/models/configs/"$model_name".toml

# 2. Train model
bonito train --training_directory $model_dir --config ../bonito/bonito/models/configs/"$model_name".toml --no-amp --epochs 15 --seed "$seed"

# 3. Get the best model on validation dataset
weightsnum=`python3 ../bonito/bonito/cli/selModel.py "$model_dir"`

# 4. Basecall
bonito basecaller ${model_dir} ../../external/bonito/Klebsiella_pneumoniae_INF032_fast5s --no-half --weights ${weightsnum} > ${result_dir}${model_name}.fastq

# 5. Test
cd test
bash sbatch_run_analysis_conv.sh "$seed"
