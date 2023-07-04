FROM ros:noetic

ARG DEBIAN_FRONTEND=noninteractive

# Set project folder
WORKDIR "/neu_nbv"
SHELL ["/bin/bash", "-c"]

# Install basic packages
RUN apt-get update && \
    apt-get install -y --no-install-recommends apt-utils nano git curl \
    python3-pip dirmngr gnupg2 ros-noetic-gazebo-ros ros-noetic-rviz ros-noetic-cv-bridge ros-noetic-gazebo-ros-pkgs && \
    rm -rf /var/lib/apt/lists/*

# Install miniconda
ENV PATH="/root/miniconda3/bin:${PATH}"
ARG PATH="/root/miniconda3/bin:${PATH}"
RUN apt-get update && apt-get install -y wget && rm -rf /var/lib/apt/lists/*
RUN wget \
    https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && mkdir /root/.conda \
    && bash Miniconda3-latest-Linux-x86_64.sh -b \
    && rm -f Miniconda3-latest-Linux-x86_64.sh \
    && echo "source /root/miniconda3/etc/profile.d/conda.sh" >> ~/.bashrc \
    && exec bash 

# Install dependencies and environment
RUN pip3 install -U catkin_tools
COPY environment.yaml .
RUN conda env create -f environment.yaml && \    
    rm -r ~/.cache/pip

# Build simulator
COPY src/simulator src/simulator
RUN . /opt/ros/noetic/setup.bash &&\
    catkin build simulator -DPYTHON_EXECUTABLE=/usr/bin/python3
RUN echo "source /opt/ros/noetic/setup.bash" >> ~/.bashrc &&\
    echo "source /neu_nbv/devel/setup.bash" >> ~/.bashrc &&\
    echo "conda activate neu-nbv" >> ~/.bashrc