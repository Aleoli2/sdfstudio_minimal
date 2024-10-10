# Define base image.
FROM ghcr.io/nerfstudio-project/nerfstudio:1.1.3

RUN apt update && apt install -y python3-pip && \
    pip install torchtyping==0.1.4

RUN apt update && apt install -y git libosmesa6
RUN git clone https://github.com/mmatl/pyopengl.git && pip install ./pyopengl
RUN pip install pyrender==0.1.45

COPY nerfstudio/ /usr/local/lib/python3.10/dist-packages/nerfstudio/
COPY scripts/* /usr/local/lib/python3.10/dist-packages/nerfstudio/scripts/
WORKDIR /workspace

#TODO install OSMesa
# CMD ns-install-cli --mode install && \
#     printf "\nalias ns-extract-mesh=\"python /usr/local/lib/python3.10/dist-packages/nerfstudio/scripts/extract_mesh.py\"" >> /root/.bashrc \
#     && /bin/bash
CMD ns-install-cli --mode install && \
    printf "\nalias ns-extract-mesh=\"python /usr/local/lib/python3.10/dist-packages/nerfstudio/scripts/extract_mesh.py\"" >> /root/.bashrc &&\
    printf "\nalias ns-render-mesh=\"python /usr/local/lib/python3.10/dist-packages/nerfstudio/scripts/render_mesh.py\"" >> /root/.bashrc \
    && /bin/bash