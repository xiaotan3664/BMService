# BMService

BMService is a framework to pipeline the whole process include pre-process, forward, post-forward running on multiple BM168x chips

## introduction

## entries

all the `xxx.cpp` files in `src/model` will be compiling seperately as an executable application, named `BMService-xxx`

## build

Get the BMService code

``` shell
git clone https://github.com/xiaotan3664/BMService.git
cd BMService
git submodule update --init --recursive
```
Download Data and models
Link: https://pan.baidu.com/s/1XazmfQ-qAa4oBZu6EoZrcQ
Password: hglm


After initializing your bmnnsdk environment, run the following commands to build:

``` shell
mkdir build && cd build
cmake ..
make -j
```

BMSerice-xxx will be generated for running
