echo "Pip installing torch=2.8.0, torchvision=0.23.0 for CUDA=$CUDA ..."
pip install torch==2.8.0 torchvision==0.23.0 \
            --index-url https://download.pytorch.org/whl/${CUDA}

echo "Pip installing lightning, torch_geometric ..."
pip install lightning==2.5.4
pip install torch_geometric
pip install torch_scatter torch_cluster torch_spline_conv \
        -f https://data.pyg.org/whl/torch-2.8.0+${CUDA}.html

echo "Pip installing poly2graph ..."
pip install poly2graph jupyterlab