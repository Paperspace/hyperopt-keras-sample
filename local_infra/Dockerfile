FROM python:3.7

RUN apt update && apt install -y graphviz

WORKDIR hyper_param

COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

COPY hyper_param/ .

ENV PYTHONPATH=$PYTHONPATH:$(pwd)

RUN mkdir -p /root/.keras/datasets && mkdir results
COPY ml_req/cifar-100-python.tar.gz /root/.keras/datasets/
