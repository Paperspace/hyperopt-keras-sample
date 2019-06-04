FROM python:3.7

RUN apt update && apt install -y graphviz

WORKDIR hyper_param

COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt
RUN sudo ln -s /usr/local/cuda/lib64/stubs/libcuda.so /usr/local/cuda/lib64/stubs/libcuda.so.1
ENV LD_LIBRARY_PATH="/usr/local/cuda/lib64/stubs/:$LD_LIBRARY_PATH"

COPY hyper_param/ .

ENV PYTHONPATH=$PYTHONPATH:$(pwd)

RUN mkdir -p /root/.keras/datasets && mkdir results

COPY ml_req/* /root/.keras/datasets/
