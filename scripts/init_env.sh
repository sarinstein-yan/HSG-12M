echo "Pip installing torch=2.6.0, torchvision=0.21.0, torchaudio=2.6.0 for CUDA="$CUDA ...
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 \
            --index-url https://download.pytorch.org/whl/${CUDA}

echo "Pip installing lightning, torch_geometric ..."
pip install lightning
pip install torch_geometric
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv \
        -f https://data.pyg.org/whl/torch-2.6.0+${CUDA}.html

echo "Pip installing poly2graph ..."
pip install poly2graph jupyterlab