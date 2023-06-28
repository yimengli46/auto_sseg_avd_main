#!/bin/sh

## Give your job a name to distinguish it from other jobs you run.
# SBATCH --job-name=run_stage_d

## General partitions: all-HiPri, bigmem-HiPri   --   (12 hour limit)
##                     all-LoPri, bigmem-LoPri, gpuq  (5 days limit)
## Restricted: CDS_q, CS_q, STATS_q, HH_q, GA_q, ES_q, COS_q  (10 day limit)
#SBATCH --partition=gpuq

## Separate output and error messages into 2 files.
## NOTE: %u=userID, %x=jobName, %N=nodeID, %j=jobID, %A=arrayID, %a=arrayTaskID
#SBATCH --output=/scratch/yli44/logs/%x-%N-%j.out  # Output file
#SBATCH --error=/scratch/yli44/logs/%x-%N-%j.err   # Error file

## Slurm can send you updates via email
#SBATCH --mail-type=BEGIN,END,FAIL         # ALL,NONE,BEGIN,END,FAIL,REQUEUE,..
#SBATCH --mail-user=yli44@gmu.edu     # Put your GMU email address here

## Specify how much memory your job needs. (2G is the default)
#SBATCH --mem=100G        # Total memory needed per task (units: K,M,G,T)

## Specify how much time your job needs. (default: see partition above)
#SBATCH --time=5-00:00   # Total time needed for job: Days-Hours:Minutes

#SBATCH --gres=gpu:1
#SBATCH --nodelist=NODE077

#SBATCH --cpus-per-task 2

## Load the relevant modules needed for the job
module load cuda/11.2
module load python/3.8.4
module load gcc/7.5.0
module load git/2.27.0.221

source /scratch/yli44/env_sam/bin/activate

## Run your program or script
#python mp_stage_a_run_Detic.py
python mp_stage_d_run_maskFormer.py