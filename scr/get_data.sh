#!/bin/sh

date
echo "Running the job now on $(hostname)"


# Check if the conda environment exists
ENV_NAME="gdc-client"
if conda env list | grep -q $ENV_NAME; then
    echo "Activating existing conda environment: $ENV_NAME"
    source activate $ENV_NAME
else
    echo "Creating new conda environment: $ENV_NAME"
    conda create -n $ENV_NAME -c bioconda gdc-client -y
    source activate $ENV_NAME
fi

# Set the base directory path
BASE_DIR="$HOME/TCGA_ML"
cd $BASE_DIR

# Create target directory if it doesn't exist
TARGET_DIR="${BASE_DIR}/data"
mkdir -p "$TARGET_DIR"
cd $TARGET_DIR

echo "Current working directory: $(pwd)"

# Path to manifest file
manifest_file="${BASE_DIR}/gdc_manifest.txt"

# Download data from GDC with gdc-client as per the manifest
gdc-client download -m $manifest_file 

# Deactivate the conda environment
conda deactivate

echo "Job finished"
echo "-------done-------"
