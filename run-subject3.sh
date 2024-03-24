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

echo "[TIME] start: $(date +"%T")"
echo "------------------------------------------------------------------"
echo "Running subject3... | Rerender + Postprocess + Eval"
echo "------------------------------------------------------------------"
echo "[SBATCH Directives]"
echo "Partition: $SLURM_JOB_PARTITION"
echo "QOS: $SLURM_JOB_QOS"
echo "Nodes: $SLURM_JOB_NUM_NODES"
echo "GPUs: $SLURM_JOB_GPUS"
echo "Memory: $SLURM_MEM_PER_NODE MB"



echo "------------------------------------------------------------------"
echo "------------------------------------------------------------------"
echo "[TIME] basic_start: $(date +"%T")"
python rerender.py --cfg config/music_keypoint_bin_1020.json -eval
echo "[TIME] basic_end  : $(date +"%T")"
echo "------------------------------------------------------------------"
echo -e "------------------------------------------------------------------\n\n\n\n\n\n\n"



echo "------------------------------------------------------------------"
echo "------------------------------------------------------------------"
echo "[TIME] dp_start: $(date +"%T")"
python rerender.py --cfg config/subject3_keypoint_dp_1530.json -eval
echo "[TIME] dp_end  : $(date +"%T")"
echo "------------------------------------------------------------------"
echo -e "------------------------------------------------------------------\n\n\n\n\n\n\n"



echo "------------------------------------------------------------------"
echo "------------------------------------------------------------------"
echo "[TIME] bin_start: $(date +"%T")"
python rerender.py --cfg config/subject3_keypoint_dp_1020.json -eval
echo "[TIME] bin_end  : $(date +"%T")"
echo "------------------------------------------------------------------"
echo -e "------------------------------------------------------------------\n\n\n\n\n\n\n"



echo "------------------------------------------------------------------"
echo "------------------------------------------------------------------"
echo "[TIME] bin_start: $(date +"%T")"
python rerender.py --cfg config/subject3_keypoint_bin_1020.json -eval
echo "[TIME] bin_end  : $(date +"%T")"
echo "------------------------------------------------------------------"
echo -e "------------------------------------------------------------------\n\n\n\n\n\n\n"
