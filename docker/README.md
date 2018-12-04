# Using torchnlp via Docker

This directory contains `Dockerfile.gpu`s to make it easy to get up and running with
torchnlp via [Docker](http://www.docker.com/).

## Installing Docker

General installation instructions are
[on the Docker site](https://docs.docker.com/installation/), or some quick links here:

* [OSX](https://www.docker.com/products/docker#/mac)
* [Ubuntu](https://docs.docker.com/engine/installation/linux/ubuntulinux/)

## Build docker image
> docker build -t torchnlp:cuda9-cudnn7-py3 -f Dockerfile.gpu .

## Run docker

For GPU support install NVidia drivers (ideally latest) and
[nvidia-docker](https://github.com/NVIDIA/nvidia-docker). Run using

> nvidia-docker run -ti torchnlp:cuda9-cudnn7-py3
