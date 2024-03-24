#!/bin/bash
#SBATCH --partition=SCSEGPU_UG
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --job-name=rerender
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --ntasks=1
#SBATCH --time=06:00:00

echo "------------------------------------------------------------------"
echo "Job # $(%j) started at $(date +"%T")"
echo "------------------------------------------------------------------"
echo "[SBATCH Directives]"
echo "Partition: $SLURM_JOB_PARTITION"
echo "QOS: $SLURM_JOB_QOS"
echo "Nodes: $SLURM_JOB_NUM_NODES"
echo "GPUs: $SLURM_JOB_GPUS"
echo "Memory: $SLURM_MEM_PER_NODE MB"
echo "------------------------------------------------------------------"
echo "------------------------------------------------------------------"

module load anaconda
eval "$(command conda 'shell.bash' 'hook' 2> /dev/null)"
conda activate fyp
# nvidia-smi
module load cuda/11.8 # echo "Cuda version: $(nvcc --version)"
module load gcc/8.5.0 # echo "GCC version: $(gcc --version)"
export CC=gccmy


echo "------------------------------------------------------------------"
echo "[TIME] subject3_rerender_rerun start: $(date +"%T")"
python rerender.py --cfg config/subject3_rerender_rerun.json -nr -nb -eval
echo "[TIME] subject3_rerender_rerun end  : $(date +"%T")"
echo "------------------------------------------------------------------"
echo -e "------------------------------------------------------------------\n\n\n\n\n\n\n"