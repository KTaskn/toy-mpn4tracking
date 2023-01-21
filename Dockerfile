FROM pytorch/pytorch:1.12.1-cuda11.3-cudnn8-runtime

RUN pip install -U pip \
    && pip install pyg-lib torch-scatter \
    torch-sparse torch-cluster torch-spline-conv \
    torch-geometric -f https://data.pyg.org/whl/torch-1.12.0+cu113.html \
    pytest pandas