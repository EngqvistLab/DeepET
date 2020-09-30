#!/bin/bash
#SBATCH -t 7-00:00:00
#SBATCH -A C3SE2020-1-14 # group to which you belong
#SBATCH -p vera  # partition (queue)
#SBATCH -n 20
#SBATCH --gres=gpu:1
#SBATCH --mail-user=gangl@chalmers.se
#SBATCH --mail-type=end                 # send mail when job ends
#SBATCH -o ../../results/tm_Jarzab_models/running_log_Scratch_RES1.txt

source /c3se/users/gangl/Vera/load_gpu
cd ../../scripts/

for i in {1..10}
do
python pre_train_simple.py \
--trainfile ../data/tm_Jarzab/cleaned_enzyme_tms_jarzab_v1_train.fasta \
--modelname RES1 \
--hyparam p_RES1_uniDist_best1 \
--tag $i \
--patience 200 \
--outdir ../results/tm_Jarzab_models/Tm_Scratch
done
conda clean --all
