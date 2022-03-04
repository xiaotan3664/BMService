#!/bin/bash

mkdir -p build
cd build
TARGET_ARCH=x86-pcie cmake ../
make -j
make install
cd ..

