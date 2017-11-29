FROM ubuntu:16.04

RUN apt-get update
RUN apt-get install -y \
    build-essential \
    bzip2 \
    cmake \
    curl \
    git \
    libboost-all-dev \
    libbz2-dev \
    libfluidsynth-dev \
    libfreetype6-dev \
    libgme-dev \
    libgtk2.0-dev \
    libjpeg-dev \
    libopenal-dev \
    libpng12-dev \
    libsdl2-dev \
    libwildmidi-dev \
    libzmq3-dev \
    nano \
    nasm \
    pkg-config \
    rsync \
    software-properties-common \
    sudo \
    tar \
    timidity \
    unzip \
    wget \
    zlib1g-dev

# Python3
RUN apt-get install -y python3-dev python3 python3-pip
RUN pip3 install pip --upgrade

# tkinter
RUN yes Y | apt-get install python3-tk

# TensorFlow
RUN pip3 install tensorflow

# Sonotano raiburari
RUN pip3 install matplotlib scipy scikit-image tqdm

#vizdoom
RUN git clone https://github.com/mwydmuch/ViZDoom ${HOME_DIR}/vizdoom
RUN pip3 install ${HOME_DIR}/vizdoom

# jupyter
RUN pip3 install jupyter

# chainer
RUN pip3 install chainer

# chainerRl for deep q learning
RUN pip3 install chainerrl

# gym for chainerrl example
RUN pip3 install gym
RUN pip3 install gym[atari]

# matplotlib
RUN pip3 install matplotlib

# pil
RUN sudo apt-get install libjpeg-dev
RUN sudo apt-get install zlib1g-dev
RUN sudo apt-get install libpng-dev
# RUN sudo apt-get install libjpeg libjpeg-dev libfreetype6 libfreetype6-dev zlib1g-dev
RUN pip3 install Pillow


# Enables X11 sharing and creates user home directory
ENV USER_NAME vizdoom
ENV HOME_DIR /home/$USER_NAME
#
# Replace HOST_UID/HOST_GUID with your user / group id (needed for X11)
ENV HOST_UID 1001
ENV HOST_GID 1001

RUN export uid=${HOST_UID} gid=${HOST_GID} && \
    mkdir -p ${HOME_DIR} && \
    echo "$USER_NAME:x:${uid}:${gid}:$USER_NAME,,,:$HOME_DIR:/bin/bash" >> /etc/passwd && \
    echo "$USER_NAME:x:${uid}:" >> /etc/group && \
    echo "$USER_NAME ALL=(ALL) NOPASSWD: ALL" > /etc/sudoers.d/$USER_NAME && \
    chmod 0440 /etc/sudoers.d/$USER_NAME && \
    chown ${uid}:${gid} -R ${HOME_DIR}

USER ${USER_NAME}
WORKDIR ${HOME_DIR}

RUN mkdir config
# python files
COPY chaintest.py .
COPY chaintest2.py .
COPY chaintest3.py .
COPY chaintrain3.py .
COPY chaintest4.py .
COPY chaintest4p4.py .
COPY chaintest5.py .
COPY chainrltuto.py .
COPY chainrltuto2.py .
COPY chaintestmodel5 ./chaintestmodel5

# doom map
COPY config ./config

# pictures
#COPY img_six.png .
#COPY img_un.png .
#COPY mnist_one.png .
#COPY mnist_eight.png .
#COPY invert_one.png .
#COPY invert_eight.png .
#COPY invert_six.png .
#COPY c_onec.png .
#COPY c_sevenc.png .
#COPY c_threec.png .
#COPY c_sixc.png .

CMD /bin/bash
