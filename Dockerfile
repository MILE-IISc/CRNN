FROM tensorflow/tensorflow

RUN apt update

# Core linux dependencies
RUN apt install -y \
        ffmpeg \
        libsm6 \
        libxext6

# Python dependencies
RUN pip --no-cache-dir install \
    opencv-python \
    tqdm

WORKDIR /
