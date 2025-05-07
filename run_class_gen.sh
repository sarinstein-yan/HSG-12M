#!/bin/bash
CLASS_DATA=/home/user/Research/sg_dataset
SAVE_DIR=/home/user/Research/sg_dataset/poly_class_dataset

for idx in $(seq 0 999); do
    echo "=== Processing class $idx ==="
    python /home/user/Research/sg_dataset/generate_poly_class.py \
        --class_idx $idx \
        --class_data "$CLASS_DATA" \
        --save_dir "$SAVE_DIR" \
        --num_partition 3 \
        --short_edge_threshold 30
done

# seq 0 1000 | parallel -j3 \
#     python /home/user/Research/sg_dataset/generate_poly_class.py \
#         --class_idx {} \
#         --class_data "$CLASS_DATA" \
#         --save_dir "$SAVE_DIR" \
#         --num_partition 3