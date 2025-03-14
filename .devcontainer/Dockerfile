FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel

RUN apt-get update \
    && apt-get install -y git gdb vim curl wget tmux zip cmake ffmpeg libsm6 libxext6

RUN pip install openai