FROM nvcr.io/nvidia/pytorch:21.03-py3

RUN apt update && \
    apt install -y bash \
                   build-essential \
                   git \
                   curl \
                   ca-certificates \
                   libsndfile1 \
                   python3 \
                   python3-pip && \
    rm -rf /var/lib/apt/lists

RUN apt update && \
    apt install -y language-pack-en && \
    locale-gen en_US.UTF-8

ENV LANG=en_US.utf8
ENV LC_ALL='en_US.utf8'

#RUN pip3 install torchaudio torchvision==0.9.1+cu111
#RUN pip3 install torch==1.8.1+cu112 torchaudio==0.8.1 torchtext==0.9.1 -f https://download.pytorch.org/whl/torch_stable.html

WORKDIR /workspace
RUN python3 -m pip install --no-cache-dir --upgrade pip && \
    python3 -m pip install --no-cache-dir jupyter

RUN git clone https://github.com/huggingface/transformers.git && \
    cd transformers && \
    pip3 install -e .

RUN pip3 install datasets && \
    pip3 install soundfile && \
    pip3 install jiwer && \
    pip3 install lang-trans && \
    pip3 install librosa && \
#    pip3 install wandb && \
    pip3 install unidecode && \
    pip3 install soundfile

#RUN git clone https://github.com/ynop/py-ctc-decode && \

RUN git clone --recursive https://github.com/parlance/ctcdecode.git && \
    cd ctcdecode && pip3 install . && cd ..

RUN pip3 install pyctcdecode
RUN apt-get -y install python-dev

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

RUN pip3 install kenlm


WORKDIR /workspace/wav2vec2

CMD ["/bin/bash"]
