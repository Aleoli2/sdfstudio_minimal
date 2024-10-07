# Define base image.
FROM ghcr.io/nerfstudio-project/nerfstudio:1.1.3

COPY nerfstudio/ /usr/local/lib/python3.10/dist-packages/nerfstudio/

WORKDIR /workspace

RUN apt update && apt install -y python3-pip && \
    pip install torchtyping==0.1.4

CMD ns-install-cli --mode install && /bin/bash