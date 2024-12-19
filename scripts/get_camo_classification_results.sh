#!/bin/bash
#SBATCH --job-name=camouflage_analysis
#SBATCH --output=camouflage_analysis.out
#SBATCH --error=camouflage_analysis.err
#SBATCH --gpus a100:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=24GB
#SBATCH --ntasks=1
#SBATCH --time=08:00:00

# Load necessary modules
module load miniforge3
module load cuda

# Navigate to the target directory
cd /home/cehrett/Projects/Trolls/baseball_tweets

# Activate the conda environment
source activate CamouflageAnalysis

# Add the project directory to the Python path
export PYTHONPATH=$PYTHONPATH:/home/cehrett/Projects/Trolls/baseball_tweets

# Execute the Python script
python camouflage_analysis/scripts/get_camo_classification_results.py
