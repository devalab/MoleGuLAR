FROM nvidia/cuda:12.3.2-cudnn9-devel-ubuntu20.04

WORKDIR /app


RUN apt-get update && apt-get install -y \
    wget \
    bzip2 \
    ca-certificates \
    curl \
    git \
    python3-dev \
    python3-pip \
    libxrender1 \
    && rm -rf /var/lib/apt/lists/*


RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && /bin/bash Miniconda3-latest-Linux-x86_64.sh -b -p /opt/conda \
    && rm Miniconda3-latest-Linux-x86_64.sh

ENV PATH=/opt/conda/bin:$PATH

COPY test.yml .
RUN conda env create -f test.yml

SHELL ["conda", "run", "-n", "test", "/bin/bash", "-c"]

RUN pip install  dgl -f https://data.dgl.ai/wheels/torch-2.3/cu121/repo.html

# Activate the environment and make it the default
RUN echo "source activate test" > ~/.bashrc
ENV PATH /opt/conda/envs/myenv/bin:$PATH

ENV MGL_TOOLS_PATH=/app/mgltools_x86_64Linux2_1.5.6
ENV AUTODOCK_GPU=/app/AutoDock-GPU
ENV OPTIMIZER_DIR=/app/Optimizer

RUN git clone https://github.com/ccsb-scripps/AutoDock-GPU && export GPU_INCLUDE_PATH=/usr/local/cuda/include && export GPU_LIBRARY_PATH=/usr/local/cuda/lib64 && cd AutoDock-GPU && make DEVICE=CUDA NUMWI=64

COPY mgltools_x86_64Linux2_1.5.6.tar_.gz .
RUN tar -xvzf mgltools_x86_64Linux2_1.5.6.tar_.gz \
    && cd mgltools_x86_64Linux2_1.5.6 \
    && ./install.sh

ENV LD_LIBRARY_PATH=${MGL_TOOLS_PATH}/lib:$LD_LIBRARY_PATH
EXPOSE 8080

COPY . /app

ENTRYPOINT [ "/bin/bash" ]
