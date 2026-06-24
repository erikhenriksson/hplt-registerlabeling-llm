#!/bin/bash -l
#SBATCH --account=project_462000999
#SBATCH --partition=standard-g
#SBATCH --gres=gpu:mi250:1
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --job-name=register
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

# --- edit these ---
INPUT=sample.jsonl
OUTPUT=sample.labeled.jsonl
# ------------------

export CONTAINER_IMAGE=/appl/local/laifs/containers/lumi-multitorch-latest.sif
module use /appl/local/laifs/modules
module load lumi-aif-singularity-bindings

export PYTHONPATH=$PWD/extra_pkgs:$PYTHONPATH

ls $PWD/extra_pkgs

singularity exec $CONTAINER_IMAGE \
    python3 4_annotate.py --input "$INPUT" --output "$OUTPUT"
