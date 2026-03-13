FROM nvidia/cuda:12.6.3-cudnn-devel-ubuntu24.04

RUN apt update && \
    DEBIAN_FRONTEND=noninteractive apt install -y git build-essential curl wget unzip pkg-config python3 libgomp1 cmake && \
    apt clean && \
    rm -rf /var/lib/apt/lists/
RUN wget https://download.pytorch.org/libtorch/cu126/libtorch-shared-with-deps-2.6.0%2Bcu126.zip && \
    unzip libtorch-shared-with-deps-2.6.0+cu126.zip
RUN wget https://github.com/microsoft/onnxruntime/releases/download/v1.24.3/onnxruntime-linux-x64-gpu-1.24.3.tgz && \
    tar xzvf onnxruntime-linux-x64-gpu-1.24.3.tgz && \
    mv onnxruntime-linux-x64-gpu-1.24.3 onnxruntime