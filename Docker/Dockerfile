ARG FROM_IMAGE_NAME=nvcr.io/nvidia/pytorch:20.10-py3
FROM ${FROM_IMAGE_NAME}

RUN apt update && apt install -y libsndfile1 && apt install -y sox && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace/jasper

# Install requirements (do this first for better caching)
COPY requirements.txt .
RUN conda install -y pyyaml==5.4.1
RUN pip install --disable-pip-version-check -U -r requirements.txt


RUN apt-get update && apt-get -y install \
    build-essential \
    cmake \
    git \
    language-pack-en \
    libsndfile1-dev \
    libsndfile1 \
    libstdc++6 \
    libprotobuf-dev \
    nano \
#    python3-libnvinfer \
#    python3-libnvinfer-dev \
#    python3-pip \
    protobuf-compiler \
    swig \
    sox \
    wget

RUN locale-gen en_US.UTF-8
ENV LANG=en_US.utf8
ENV LC_ALL='en_US.utf8'

#RUN git clone https://github.com/kpu/kenlm && \
#    wget http://www.openfst.org/twiki/pub/FST/FstDownload/openfst-1.6.3.tar.gz && \
#    tar -xzvf openfst-1.6.3.tar.gz && \
#    git clone https://github.com/progschj/ThreadPool.git && \
#    python3 setup.py install --num_processes 8


#RUN git clone --recursive https://github.com/parlance/ctcdecode.git
#RUN cd ctcdecode && pip3 install .

RUN pip3 install pyctcdecode


RUN pip3 install kenlm

WORKDIR /workspace/jasper
COPY . .



#FROM nvcr.io/nvidia/pytorch:21.03-py3
#ARG DEBIAN_FRONTEND=noninteractive
#
#RUN apt update && \
#    apt install -y bash \
#                   build-essential \
#                   git \
#                   cmake \
#                   curl \
#                   ca-certificates \
#                   libsndfile1 \
#                   nano \
#                   python3 \
#                   python3-pip \
#                   python-dev && \
#    rm -rf /var/lib/apt/lists
#
#RUN apt update && \
#    apt install -y language-pack-en && \
#    locale-gen en_US.UTF-8
#
#ENV LANG=en_US.utf8
#ENV LC_ALL='en_US.utf8'
#
#
#WORKDIR /workspace
#RUN python3 -m pip install --no-cache-dir --upgrade pip && \
#    python3 -m pip install --no-cache-dir jupyter
#
#RUN git clone https://github.com/huggingface/transformers.git && \
#    cd transformers && \
#    pip3 install -e .
#
#RUN pip3 install datasets soundfile jiwer kenlm lang-trans librosa pyctcdecode unidecode soundfile
#
#RUN git clone --recursive https://github.com/parlance/ctcdecode.git && \
#    cd ctcdecode && pip3 install . && cd ..
#
#WORKDIR /workspace/wav2vec2
#
#CMD ["/bin/bash"]
