prefix=mar38v_hac
model_dir=../../../models/"$prefix"
result_dir=../../result/bonito/
raw_data=${prefix}.fastq

# 1. Delete useless lines in the result file
tail -n +14 ../${result_dir}$raw_data > ../${result_dir}tail_$raw_data

# 2. fix read names (necessary for consensus accuracy evaluation)
python3 analysis/fix_read_names.py ../${result_dir}tail_$raw_data read_id_to_fast5.tab > ../${result_dir}fix_$raw_data

# 3. Evaluate read and consensus accuracy
bash analysis/analysis.sh ../${result_dir}fix_$raw_data
python3 analysis/stat_result.py fix_$raw_data
