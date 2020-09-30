#!/bin/bash
#SBATCH -A C3SE2020-1-14          # NOTE: change to your own project!
#SBATCH -N 1
#SBATCH -n 32
#SBATCH -p vera
#SBATCH -o out_ifeature.log       # %N for name of 1st allocated node
#SBATCH -t 7-00:00:00                      # walltime limit
#SBATCH --mail-user=gangl@chalmers.se
#SBATCH --mail-type=end                 # send mail when job ends

module load GCC/6.4.0-2.28  CUDA/9.1.85  OpenMPI/2.1.2
module load Python/3.6.7
source /c3se/NOBACKUP/users/gangl/Tools/my_python3_vera/bin/activate
cd ../../classical_models/

python evaluate_on_classical_models.py \
--infile ../data/tm_Jarzab/tm_iFeatures.csv \
--testfile ../data/tm_Jarzab/cleaned_enzyme_tms_jarzab_v1_test.fasta \
--outfile ../results/tm_Jarzab_models/iFeatures_score.csv