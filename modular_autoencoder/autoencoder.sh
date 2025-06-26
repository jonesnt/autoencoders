#!/bin/bash

#########################################################################
# ADVANCED SLURM JOB EXAMPLE
# This shows more sophisticated job submission features
# Author: USC Binev Lab  
# Description: GPU-accelerated deep learning job with advanced features
#########################################################################

# Job name with informative naming convention
#SBATCH --job-name=test_autoencoder

# Account (always required)
#SBATCH --account=binev-lab

# Longer runtime for training jobs
#SBATCH --time=04:00:00

# Single task but multiple CPUs for data loading
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8

# More memory for large datasets
#SBATCH --mem=32G

# Request 1 GPU (this is the key for ML jobs!)
#SBATCH --gres=gpu:1

# Partition selection (optional - SLURM will choose if not specified)
#SBATCH --partition=binev-H200

# Output files with more descriptive names
#SBATCH --output=logs/autoencoder_%j_%x.out
#SBATCH --error=logs/autoencoder_%j_%x.err

# Email notifications for long jobs
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=JONESNT@email.sc.edu

# Quality of Service (QoS) - controls priority
# #SBATCH --qos=normal

#########################################################################
# ENVIRONMENT SETUP
#########################################################################

# Create logs directory if it doesn't exist
mkdir -p logs

# Print comprehensive job information
echo "==========================================="
echo "SLURM JOB INFORMATION"
echo "==========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURM_NODELIST"
echo "Task ID: $SLURM_PROCID"
echo "CPUs per task: $SLURM_CPUS_PER_TASK"
echo "Memory: $SLURM_MEM_PER_NODE MB"
echo "Working directory: $(pwd)"
echo "Started at: $(date)"

# If using job arrays, show array information
if [ ! -z "$SLURM_ARRAY_TASK_ID" ]; then
    echo "Array Job ID: $SLURM_ARRAY_JOB_ID"
    echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
fi

echo "==========================================="

# Load modules
echo "Loading modules..."
module load python3/anaconda/3.12

# You might need additional modules for specific software
# module load cuda/11.8
# module load gcc/9.3.0

# Activate environment
echo "Activating Python environment..."
source activate /work/binev-lab/venvs/binev-env/

# Verify GPU access
echo "==========================================="
echo "GPU INFORMATION"
echo "==========================================="
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi
    echo "CUDA Version: $(nvcc --version | grep 'release' | awk '{print $6}' | cut -c2-)"
else
    echo "Warning: nvidia-smi not found. GPU may not be available."
fi

# Check PyTorch GPU access
echo "Testing PyTorch GPU access..."
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU count: {torch.cuda.device_count()}'); print(f'Current device: {torch.cuda.current_device() if torch.cuda.is_available() else \"CPU only\"}')"

echo "==========================================="

#########################################################################
# EXPERIMENT PARAMETERS
#########################################################################

# Set experiment parameters (makes it easy to modify)
BATCH_SIZE=32
LEARNING_RATE=0.001
EPOCHS=5000
LATENT_DIM=16

echo "EXPERIMENT PARAMETERS:"
echo "Batch size: $BATCH_SIZE"
echo "Learning rate: $LEARNING_RATE"
echo "Epochs: $EPOCHS"
echo "Latent dimension: $LATENT_DIM"
echo "==========================================="

#########################################################################
# DATA PREPARATION
#########################################################################

# Set paths
DATA_DIR="/work/binev-lab/shared_data/microscopy_images"
OUTPUT_DIR="./results/exp_$(date +%Y%m%d_%H%M%S)_job${SLURM_JOB_ID}"
CHECKPOINT_DIR="$OUTPUT_DIR/checkpoints"

# Create output directories
mkdir -p $OUTPUT_DIR
mkdir -p $CHECKPOINT_DIR

echo "Data directory: $DATA_DIR"
echo "Output directory: $OUTPUT_DIR"

# Check if data exists
if [ ! -d "$DATA_DIR" ]; then
    echo "Warning: Data directory $DATA_DIR not found. Using synthetic data."
fi

#########################################################################
# MAIN COMPUTATION
#########################################################################

echo "Starting main computation..."

# Run the Python script with parameters
python training_manager.py \
    --batch_size $BATCH_SIZE \
    --learning_rate $LEARNING_RATE \
    --epochs $EPOCHS \
    --latent_dim $LATENT_DIM \
    --data_dir $DATA_DIR \
    --output_dir $OUTPUT_DIR \
    --checkpoint_dir $CHECKPOINT_DIR \
    --job_id $SLURM_JOB_ID

# Capture exit code
EXIT_CODE=$?

#########################################################################
# POST-PROCESSING
#########################################################################

if [ $EXIT_CODE -eq 0 ]; then
    echo "Training completed successfully!"
    
    # Copy important results to a permanent location
    cp $OUTPUT_DIR/*.png /work/binev-lab/shared_results/
    cp $OUTPUT_DIR/*.pth /work/binev-lab/shared_results/
    
    echo "Results copied to shared directory."
else
    echo "Training failed with exit code: $EXIT_CODE"
    
    # Save debug information
    echo "Saving debug information..."
    env > $OUTPUT_DIR/environment.txt
    cp $0 $OUTPUT_DIR/job_script.sh
fi

#########################################################################
# CLEANUP AND REPORTING
#########################################################################

# Clean up temporary files if needed
# rm -f /tmp/temp_data_${SLURM_JOB_ID}*

# Print job statistics
echo "==========================================="
echo "JOB COMPLETION INFORMATION"
echo "==========================================="
echo "Job completed at: $(date)"
echo "Exit code: $EXIT_CODE"
echo "Output directory: $OUTPUT_DIR"

# If available, print job efficiency information
if command -v seff &> /dev/null; then
    echo "Job efficiency:"
    seff $SLURM_JOB_ID
fi

echo "==========================================="

# Exit with the same code as the Python script
exit $EXIT_CODE

#########################################################################
# ADVANCED FEATURES EXPLAINED:
#
# 1. GPU REQUEST: --gres=gpu:1 requests one GPU
# 2. JOB ARRAYS: Run multiple parameter combinations automatically
# 3. ORGANIZED OUTPUTS: Creates timestamped directories for each run
# 4. PARAMETER PASSING: Passes SLURM variables to Python script
# 5. ERROR HANDLING: Captures and reports exit codes properly
# 6. ENVIRONMENT LOGGING: Saves environment info for debugging
# 7. AUTOMATIC CLEANUP: Removes temporary files
# 8. RESULT ARCHIVING: Copies important results to shared storage
#
# SUBMIT EXAMPLES:
# sbatch advanced_job_example.sh                    # Single job
# sbatch --array=1-5 advanced_job_example.sh       # 5 parameter sweeps
# sbatch --time=08:00:00 advanced_job_example.sh   # Override time limit
#
# MONITORING:
# squeue -u $USER                    # Check your jobs
# scontrol show job JOB_ID          # Detailed job info
# tail -f logs/autoencoder_*.out     # Watch live output
# seff JOB_ID                       # Job efficiency after completion
#########################################################################