# Base image with PyTorch, CUDA 12.4, and cuDNN 9
FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Set the working directory
WORKDIR /home/InSpect

# Install necessary system packages
RUN apt-get update && apt-get install -y \
    python3-pip \
    git \
    wget \
    vim \
    && apt-get clean

# Upgrade pip
RUN pip install --upgrade pip

# Install Python dependencies
COPY requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt

# Copy the InSpect project into the container
COPY . /home/InSpect/

# Default command
CMD ["bash"]
