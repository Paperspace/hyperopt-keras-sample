FROM tensorflow/tensorflow:1.13.1-gpu-py3

RUN apt update && apt install -y graphviz

WORKDIR hyper_param

COPY requirements.txt requirements.txt
RUN ln -f -s /usr/local/cuda-10.0/compat/libcuda.so.1 /usr/lib/x86_64-linux-gnu/libcuda.so.1 && \
    pip3 install -r requirements.txt

COPY hyper_param/ .

ENV PYTHONPATH=$PYTHONPATH:$(pwd)
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-10.0/compat/

RUN mkdir -p /root/.keras/datasets && mkdir results

COPY ml_req/* /root/.keras/datasets/
