for i in 23 123 345 1234 9 2323 927 48 100 420 60 320 7767 51 42
do
    echo conv with seed $i
    sbatch sbatch_run_train_basecall_conv.sh $i
    echo markonv with seed $i
    sbatch sbatch_run_train_basecall_markonv.sh $i
done
