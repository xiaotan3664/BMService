#!/bin/bash

mkdir -p build
cd build
cmake ../
make -j
cp libbmservice.so ../python/bmservice/lib
cd ..

