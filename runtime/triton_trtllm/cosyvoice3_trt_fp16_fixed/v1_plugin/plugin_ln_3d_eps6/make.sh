TRT_IN_DIRS=/usr/local/tensorrt/include/
TRT_LIB_DIRS=/usr/local/tensorrt/lib/
GPU_CC=89

rm -rf ./build/
mkdir build && cd build
cmake -DGPU_CC=${GPU_CC} -DTRT_IN_DIRS=${TRT_IN_DIRS} -DTRT_LIB_DIRS=${TRT_LIB_DIRS} ..
make -j20
