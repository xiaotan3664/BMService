#!/bin/bash

mkdir -p build
cd build
cmake ../
make -j
make install
cd ..

