FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04
RUN apt-get update && apt install -yy google-perftools cmake ninja-build curl
RUN curl -LO https://r2.drysys.workers.dev/torch/torch-2.1.0a0+git3af011b-cp311-cp311-linux_x86_64.whl
WORKDIR /src
COPY . .
#RUN pip3 install ./torch-2.1.0a0+git3af011b-cp311-cp311-linux_x86_64.whl
RUN apt install python3.11 libpython3.11-dev
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.11
RUN pip install /torch-2.1.0a0+git3af011b-cp311-cp311-linux_x86_64.whl
RUN TORCH_CUDA_ARCH_LIST="8.0;8.6" python3.11 -c 'import cuda_ext'

