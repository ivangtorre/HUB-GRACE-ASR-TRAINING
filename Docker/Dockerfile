FROM nvidia/cuda:10.2-cudnn7-devel-ubuntu18.04
LABEL maintainer="Hugging Face"
LABEL repository="transformers"

RUN apt update && \
    apt install -y bash \
                   build-essential \
                   git \
                   curl \
                   ca-certificates \
                   python3 \
                   python3-pip && \
    rm -rf /var/lib/apt/lists

RUN python3 -m pip install --no-cache-dir --upgrade pip && \
    python3 -m pip install --no-cache-dir \
    mkl \
    torch

RUN git clone https://github.com/NVIDIA/apex
RUN cd apex && \
    python3 setup.py install && \
    pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./

WORKDIR /workspace

RUN git clone https://github.com/huggingface/transformers.git && cd transformers && python3 -m pip install --no-cache-dir .

RUN pip3 install datasets

CMD ["/bin/bash"]

#ARG FROM_IMAGE_NAME=nvcr.io/nvidia/pytorch:20.03-py3
#FROM ${FROM_IMAGE_NAME}
#ARG DEBIAN_FRONTEND=noninteractive
#
#
#RUN apt-get update && apt-get -y install \
#    build-essential \
#    cmake \
#    git \
#    language-pack-en \
#    libsndfile1-dev \
#    libsndfile1 \
#    libstdc++6 \
#    libprotobuf-dev \
#    nano \
#    protobuf-compiler \
#    swig \
#    sox \
#    wget
#
#RUN locale-gen en_US.UTF-8
#ENV LANG=en_US.utf8
#ENV LC_ALL='en_US.utf8'
#
#WORKDIR /workspace
#
#RUN git clone https://github.com/huggingface/transformers.git && cd transformers && pip install -e .
#
## Install requirements (do this first for better caching)
#COPY requirements.txt .
##RUN conda install -y pyyaml==5.4.1
##RUN conda install -y pyyaml==5.4.1
#
##RUN pip install --disable-pip-version-check -U -r requirements.txt
#RUN pip install -U -r requirements.txt
#
##WORKDIR /workspace/wav2vec2

WORKDIR /workspace

CMD ["/bin/bash"]
