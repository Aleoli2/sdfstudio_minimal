This repository contains the files to implement some of the sdfstudio algorithms (for now, BakedSDF) in nerfstudio 1.1.3, and a dockerfile to construct it from the image dromni/nerfstudio:1.1.3. Follew the instructions of [nerfstudio](https://docs.nerf.studio/) and [sdfstudio](https://github.com/autonomousvision/sdfstudio) for using it.

Build the Dockerfile with:\
`docker build --build-arg USER_ID=$(id -u) -t sdfstudio_minimal .`