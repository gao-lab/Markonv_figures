#!/bin/bash

model_name=dna_r9.4.1@v3.3
model_dir=../../models/"$model_name"
test_dir=../bonito/test/
result_dir=../../result/bonito/

# 1. View model
bonito view ../bonito/bonito/models/configs/"$model_name".toml

# 2. Train model
bonito train --training_directory $model_dir --config ../bonito/bonito/models/configs/"$model_name".toml --no-amp --epochs 15

# 3. Get the best model on validation dataset
weightsnum=`python3 ../bonito/bonito/cli/selModel.py "$model_dir"`

# 4. Basecall
/usr/bin/time --verbose -o ${result_dir}${model_name}.time bonito basecaller ${model_dir} ../../external/bonito/Klebsiella_pneumoniae_INF032_fast5s --no-half --weights ${weightsnum} > ${result_dir}${model_name}.fastq
