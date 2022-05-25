ARG FROM_IMAGE_NAME=nvcr.io/nvidia/pytorch:20.12-py3
FROM ${FROM_IMAGE_NAME}
ARG DEBIAN_FRONTEND=noninteractive

RUN pip3 install torch==1.8.1+cu111 torchaudio==0.8.1 torchtext==0.9.1 torchvision==0.9.1 -f https://download.pytorch.org/whl/torch_stable.html

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
    protobuf-compiler \
    swig \
    sox \
    wget

RUN locale-gen en_US.UTF-8
ENV LANG=en_US.utf8
ENV LC_ALL='en_US.utf8'

WORKDIR /workspace

RUN git clone -b v4.13.0 https://github.com/huggingface/transformers && cd transformers && pip install -e .

# Install requirements (do this first for better caching)
RUN apt-get update && apt-get -y install python-dev
RUN pip install ipdb librosa pandas pyyaml soundfile sox tqdm wrapt datasets jiwer lang-trans pyctcdecode unidecode
RUN pip install https://github.com/kpu/kenlm/archive/master.zip
RUN pip3 install datasets

#WORKDIR /workspace/wav2vec2

CMD ["/bin/bash"]
