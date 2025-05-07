#!/bin/bash
# CLASS_DATA=/home/users/nus/e1184420/research/NHSG-12M/poly_class_dataset/ParamTable.npz
CLASS_DATA=/home/users/nus/e1184420/scratch/nhsg12m/ParamTable.npz
SAVE_DIR=/home/users/nus/e1184420/scratch/nhsg12m/poly_class_dataset

for idx in $(seq 1001 1223); do
    echo "=== Processing class $idx ==="
    python /home/users/nus/e1184420/research/NHSG-12M/generate_poly_class.py \
        --class_idx $idx \
        --class_data "$CLASS_DATA" \
        --save_dir "$SAVE_DIR" \
        --num_partition 2 \
        --short_edge_threshold 30
done

for idx in $(seq 1225 1379); do
    echo "=== Processing class $idx ==="
    python /home/users/nus/e1184420/research/NHSG-12M/generate_poly_class.py \
        --class_idx $idx \
        --class_data "$CLASS_DATA" \
        --save_dir "$SAVE_DIR" \
        --num_partition 2 \
        --short_edge_threshold 30
done

# seq 0 1000 | parallel -j3 \
#     python /home/user/research/sg_dataset/NHSG-12M/generate_poly_class.py \
#         --class_idx {} \
#         --class_data "$CLASS_DATA" \
#         --save_dir "$SAVE_DIR" \
#         --num_partition 3