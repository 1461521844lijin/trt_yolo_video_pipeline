#!/bin/bash
# This file contains documentation snippets for Linux installation tutorial
if [ "$1" = "--check" ] ; then
sudo()
{
    command $@
}
fi

# [body]
# Install minimal prerequisites (Ubuntu 18.04 as reference)
sudo apt update && sudo apt install -y cmake g++ wget unzip

# Download and unpack sources
wget -O opencv.zip https://github.com/opencv/opencv/archive/4.x.zip
unzip opencv.zip

# Create build directory
mkdir -p build && cd build

# Configure
cmake  ../opencv-4.x

# Build
cmake --build .
# [body]
