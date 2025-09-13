#!/bin/bash

# Usage: ./run_qubo.sh <path> <T> <slope> <dmax>
# Example: ./run_qubo.sh data/qubo/exact/40 0.1 0.01 4

# Check arguments
if [ "$#" -ne 4 ]; then
    echo "Usage: $0 <path> <T> <slope> <dmax>"
    exit 1
fi

DATA_PATH=$1
T=$2
slope=$3
dmax=$4


# Iterate over each JSON file in the folder
for json_file in "$DATA_PATH"/*.json; do
    filename=$(basename "$json_file")
    echo "Running on $filename ..."
    python3 qubo.py \
        --double_precision \
        --sin_lambda \
        --path="$DATA_PATH" \
        --filename="$filename" \
        --T="$T" \
        --nitime=1 \
        --slope="$slope" \
        --dmax="$dmax"
done
