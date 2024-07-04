#!/bin/bash

# Define the total number of records and number of GPUs
TOTAL_RECORDS=444681  # Adjust based on your dataset size
GPUS=4
RECORDS_PER_GPU=$(($TOTAL_RECORDS / $GPUS))

# Loop over each GPU and run the script
for i in $(seq 0 $(($GPUS - 1)))
do
    START=$(($i * $RECORDS_PER_GPU))
    END=$(($START + $RECORDS_PER_GPU - 1))
    
    # Last GPU takes any remaining records
    if [ $i -eq $(($GPUS - 1)) ]; then
        END=$TOTAL_RECORDS
    fi
    
    echo "Running on GPU $i, processing records from $START to $END"
    CUDA_VISIBLE_DEVICES=$i python convert_s2tokens.py --start $START --end $END --part_id $i &
done

wait
echo "All processing complete."
