FROM python:3.8-bullseye

RUN apt-get update && \
    apt-get -y install sudo && \
    pip install -U pip setuptools wheel && \
    useradd -m -r fabioklr && \
    adduser fabioklr sudo
    
WORKDIR /home/fabioklr/masterthesis

COPY requirements.txt ./
RUN pip install -r requirements.txt

COPY . .

RUN chown -R fabioklr .

ARG GIT_HASH
ENV GIT_HASH=${GIT_HASH:-dev}

USER fabioklr

RUN git config --global init.defaultBranch main &&\
    git init &&\
    git remote add origin https://github.com/fabioklr/master_thesis