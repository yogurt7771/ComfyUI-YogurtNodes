#!/bin/bash
name=$(basename "$PWD")
apt-get update && DEBIAN_FRONTEND=noninteractive apt-get -y install --no-install-recommends \
    build-essential \
    cmake \
    psmisc \
    iproute2 \
    libjpeg-dev \
    libpng-dev \
    tmux \
    libgl1-mesa-glx \
    x11-apps \
    xorg \
    xterm

source activate base
pip install isort flake8 black ipython
pip install -r requirements.txt
