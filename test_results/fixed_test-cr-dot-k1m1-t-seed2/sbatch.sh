#!/bin/bash
#SBATCH -p gpu_8
# #SBATCH -A 
#SBATCH -J fixed_test-cr-dot-k1m1-t-seed2
#SBATCH --array 0-0%0

# Please use the complete path details :
#SBATCH -D /pfs/data5/home/kit/stud/uprnr/imgrerl
#SBATCH -o /home/kit/stud/uprnr/imgrerl/test_results/fixed_test-cr-dot-k1m1-t-seed2/slurmlog/out_%A_%a.log
#SBATCH -e /home/kit/stud/uprnr/imgrerl/test_results/fixed_test-cr-dot-k1m1-t-seed2/slurmlog/err_%A_%a.log

# Cluster Settings
#SBATCH -n 1         # Number of tasks
#SBATCH -c 4  # Number of cores per task
#SBATCH -t 10:0:00             # 1:00:00 Hours, minutes and seconds, or '#SBATCH -t 10' - only minutes

#SBATCH --gres gpu:1
#SBATCH --mem-per-cpu 15000
# -------------------------------

# Activate the virtualenv / conda environment



# Export Pythonpath


# Additional Instructions from CONFIG.yml


python3 experiments/run_cw_exp.py /home/kit/stud/uprnr/imgrerl/experiments/our_configs/fixed_exp_config-cr-dot-k1m1-t-seed2.yml -j $SLURM_ARRAY_TASK_ID 

# THIS WAS BUILT FROM THE DEFAULLT SBATCH TEMPLATE