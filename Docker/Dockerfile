ARG FROM_IMAGE_NAME=huggingface/transformers-pytorch-gpu
FROM ${FROM_IMAGE_NAME}


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
