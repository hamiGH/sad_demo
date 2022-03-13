FROM nvidia/cuda:11.0.3-base-ubuntu18.04
EXPOSE 5005
ENV USER=root

ENV TZ=America/New_York 

RUN apt-get update && apt-get install -y tzdata

# Install some basic utilities
RUN apt-get update && apt-get install -y \
    curl \
    ca-certificates \
    sudo \
    git \
    bzip2 \
    libx11-6 \
    build-essential -y \
    ffmpeg -y

# Install Miniconda
ENV PATH="/root/miniconda3/bin:${PATH}"
ARG PATH="/root/miniconda3/bin:${PATH}"

RUN apt-get install -y wget && rm -rf /var/lib/apt/lists/*

RUN wget \
    https://repo.anaconda.com/miniconda/Miniconda3-py38_4.10.3-Linux-x86_64.sh \
    && mkdir /root/.conda \
    && bash Miniconda3-py38_4.10.3-Linux-x86_64.sh -b \
    && rm -f Miniconda3-py38_4.10.3-Linux-x86_64.sh

# Install SPU Requirements
RUN mkdir -p spu
COPY . ./spu
RUN pip install -r spu/requirements.txt


# Install Pytorch
RUN wget -c https://download.pytorch.org/whl/cu110/torch-1.7.0%2Bcu110-cp38-cp38-linux_x86_64.whl \
     && pip install torch-1.7.0+cu110-cp38-cp38-linux_x86_64.whl \
     &&  rm -rf torch-1.7.0+cu110-cp38-cp38-linux_x86_64.whl
RUN wget -c https://download.pytorch.org/whl/torchaudio-0.7.0-cp38-cp38-linux_x86_64.whl \
    && pip install torchaudio-0.7.0-cp38-cp38-linux_x86_64.whl \
    && rm -rf torchaudio-0.7.0-cp38-cp38-linux_x86_64.whl


CMD ["python", "spu/server.py"]
