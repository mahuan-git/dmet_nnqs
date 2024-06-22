#!/bin/sh
#An example for serial job.
#SBATCH -J test
#SBATCH -o job-%j.log
#SBATCH -e job-%j.err
#SBATCH -p shh
##SBATCH -p GPU-8A100 
##SBATCH --qos=gpu_8a100
#SBATCH -N 1 -n 1
#SBATCH --gres=gpu:a100:1
##SBATCH --ntasks=1 --cpus-per-task=4
##SBATCH --ntasks=2
echo Time is `date`
echo Directory is $PWD
echo This job runs on the following nodes:
echo $SLURM_JOB_NODELIST
echo This job has allocated 1 cpu core.
. /etc/profile.d/modules.sh
#module load icc/2022.1.0
which python
python -u 01-dmet-hchain.py >runlog


