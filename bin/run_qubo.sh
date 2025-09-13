#!/bin/bash

# Usage:
#   ./run_qubo.sh --path=data/rudolf/instances/exact/40 --T=0.1 --slope=0.01 --dmax=16 --nitime=1 --double_precision --sin_lambda
#
# The script will run qubo.py for every .json file in the provided path

# Collect all arguments
ARGS=("$@")

# Extract path argument (needed to find JSON files)
DATA_PATH=""
for arg in "$@"; do
    case $arg in
        --path=*)
            DATA_PATH="${arg#*=}"
            ;;
    esac
done

# Check that path is provided
if [ -z "$DATA_PATH" ]; then
    echo "Error: --path must be provided."
    exit 1
fi

# Iterate over each JSON file in the folder
for json_file in "$DATA_PATH"/*.json; do
    filename=$(basename "$json_file")

    CMD=(python3 qubo.py "${ARGS[@]}" --filename="$filename")

    # Print the full command
    echo "${CMD[@]}"

    # Run it
    "${CMD[@]}"
done
