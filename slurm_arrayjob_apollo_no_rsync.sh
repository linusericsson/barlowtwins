#!/bin/bash

# Location for stdout log - see https://slurm.schedmd.com/sbatch.html#lbAH
#SBATCH --output=/home/%u/projects/slurm_logs/slurm-%A_%a.out

# Location for stderr log - see https://slurm.schedmd.com/sbatch.html#lbAH
#SBATCH --error=/home/%u/projects/slurm_logs/slurm-%A_%a.out

# Maximum number of nodes to use for the job
#SBATCH --nodes=1

# Generic resources to use - typically you'll want gpu:n to get n gpus
#SBATCH --gres=gpu:1

# Megabytes of RAM required. Check `cluster-status` for node configurations
#SBATCH --mem=32G

# Number of CPUs to use. Check `cluster-status` for node configurations
#SBATCH --cpus-per-task=7

# Maximum time for the job to run, format: days-hours:minutes:seconds
#SBATCH --time=14-00:00:00

# Partition of the cluster to pick nodes from (check `sinfo`)
#SBATCH --partition=apollo

# Any nodes to exclude from selection
# #SBATCH --exclude=charles[11-19]

# Nodes to allow
# #SBATCH --nodelist=apollo1

# =====================
# Logging information
# =====================

# slurm info - more at https://slurm.schedmd.com/sbatch.html#lbAJ
echo "Job running on ${SLURM_JOB_NODELIST}"

dt=$(date '+%d/%m/%Y %H:%M:%S')
echo "Job started: $dt"

# ===================
# Environment setup
# ===================

echo "Setting up bash enviroment"

# Make available all commands on $PATH as on headnode
source ~/.bashrc

# Make script bail out after first error
set -e

# Make your own folder on the node's scratch disk
# N.B. disk could be at /disk/scratch_big, or /disk/scratch_fast. Check
# yourself using an interactive session, or check the docs:
#     http://computing.help.inf.ed.ac.uk/cluster-computing
SCRATCH_DISK=/disk/scratch
SCRATCH_HOME=${SCRATCH_DISK}/${USER}
mkdir -p ${SCRATCH_HOME}
export DATA_ROOT=${SCRATCH_HOME}/projects/data
#export EXP_ROOT=${SCRATCH_HOME}/projects/exps
export EXP_ROOT=/home/${USER}/projects/
mkdir -p ${DATA_ROOT}
mkdir -p ${EXP_ROOT}

echo $DATA_ROOT
echo $EXP_ROOT


# =================================
# Move input data to scratch disk
# =================================
# Move data from a source location, probably on the distributed filesystem
# (DFS), to the scratch space on the selected node. Your code should read and
# write data on the scratch space attached directly to the compute node (i.e.
# not distributed), *not* the DFS. Writing/reading from the DFS is extremely
# slow because the data must stay consistent on *all* nodes. This constraint
# results in much network traffic and waiting time for you!
#
# This example assumes you have a folder containing all your input data on the
# DFS, and it copies all that data file to the scratch space, and unzips it. 
#
# For more guidelines about moving files between the distributed filesystem and
# the scratch space on the nodes, see:
#     http://computing.help.inf.ed.ac.uk/cluster-tips

#echo "Moving input data to the compute node's scratch space: $SCRATCH_DISK"

# input data directory path on the DFS
src_path=/home/${USER}/projects/data

# input data directory path on the scratch disk of the node
dest_path=${DATA_ROOT}
mkdir -p ${dest_path}  # make it if required

# Important notes about rsync:
# * the --compress option is going to compress the data before transfer to send
#   as a stream. THIS IS IMPORTANT - transferring many files is very very slow
# * the final slash at the end of ${src_path}/ is important if you want to send
#   its contents, rather than the directory itself. For example, without a
#   final slash here, we would create an extra directory at the destination:
#       ${SCRATCH_HOME}/project_name/data/input/input
# * for more about the (endless) rsync options, see the docs:
#       https://download.samba.org/pub/rsync/rsync.html

#rsync --archive --update --compress --progress ${src_path}/ ${dest_path}

# Activate your conda environment
CONDA_ENV_NAME=ssl
echo "Activating conda environment: ${CONDA_ENV_NAME}"
conda activate ${CONDA_ENV_NAME}

# Move into the code root folder
cd ~/projects/barlowtwins

# ==============================
# Finally, run the experiment!
# ==============================
# Read line number ${SLURM_ARRAY_TASK_ID} from the experiment file and run it
# ${SLURM_ARRAY_TASK_ID} is simply the number of the job within the array. If
# you execute `sbatch --array=1:100 ...` the jobs will get numbers 1 to 100
# inclusive.

experiment_text_file=$1
COMMAND="`sed \"${SLURM_ARRAY_TASK_ID}q;d\" ${experiment_text_file}`"
echo "Running provided command: ${COMMAND}"
eval "${COMMAND}"
echo "Command ran successfully!"


# ======================================
# Move output data from scratch to DFS
# ======================================
# This presumes your command wrote data to some known directory. In this
# example, send it back to the DFS with rsync

#echo "Moving output data back to DFS"

src_path=${EXP_ROOT}
dest_path=/home/${USER}/projects/exps
#rsync --archive --update --compress --progress ${src_path}/ ${dest_path}


# =========================
# Post experiment logging
# =========================
echo ""
echo "============"
echo "job finished successfully"
dt=$(date '+%d/%m/%Y %H:%M:%S')
echo "Job finished: $dt"
