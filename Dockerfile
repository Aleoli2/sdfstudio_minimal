# Define base image.
FROM ghcr.io/nerfstudio-project/nerfstudio:1.1.3

COPY nerfstudio/ /usr/local/lib/python3.10/dist-packages/nerfstudio/
COPY scripts/* /usr/local/lib/python3.10/dist-packages/nerfstudio/scripts/
WORKDIR /workspace

RUN apt update && apt install -y python3-pip && \
    pip install torchtyping==0.1.4

CMD ns-install-cli --mode install && \
    printf "\nalias ns-extract-mesh=\"python /usr/local/lib/python3.10/dist-packages/nerfstudio/scripts/extract_mesh.py\"" >> /root/.bashrc \
    && /bin/bash