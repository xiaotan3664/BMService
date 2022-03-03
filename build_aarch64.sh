#!/bin/bash

mkdir -p build_aarch64
cd build_aarch64
TARGET_ARCH=arm-pcie cmake ../
make -j4
cp libbmservice.so ../python/bmservice/lib
cd ..

