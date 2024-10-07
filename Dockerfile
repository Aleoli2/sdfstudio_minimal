# Define base image.
FROM dromni/nerfstudio:1.1.3

ARG USER_ID
# Create non root user and setup environment.
RUN usermod -u ${USER_ID} user

# Switch to new user.
USER user
COPY * /home/user/nerfstudio/nerfstudio

WORKDIR /workspace

CMD ns-install-cli && /bin/bash