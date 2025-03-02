FROM nvidia/cuda:11.7.1-cudnn8-devel-ubuntu20.04

# prepare system packages
ENV DEBIAN_FRONTEND=noninteractive
RUN apt update && apt -y install python3-pip git ffmpeg libsm6 libxext6 python-is-python3

# specify gpu device ordering
ENV CUDA_DEVICE_ORDER='PCI_BUS_ID' 

# install python packages
RUN pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 --extra-index-url https://download.pytorch.org/whl/cu117

# downgrade pip temporarily (neeeded for openmim)
RUN python3 -m pip install pip~=19.0
RUN pip install openmim==0.3.9 
RUN mim install mmcv-full==1.7.2 mmdet==2.28.1
RUN python3 -m pip install pip~=23.0

COPY requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt