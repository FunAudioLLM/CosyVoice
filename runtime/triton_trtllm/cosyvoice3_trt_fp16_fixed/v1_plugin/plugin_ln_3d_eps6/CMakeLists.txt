cmake_minimum_required(VERSION 3.5 FATAL_ERROR)
 
project(trt_plugin)

option(CUDA_USE_STATIC_CUDA_RUNTIME OFF)
set(CMAKE_CXX_STANDARD 17)
add_definitions(-std=c++17)
set(CMAKE_BUILD_TYPE Release)#Debug#Release
#set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/workspace)
#set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/workspace)

find_package(CUDA REQUIRED)

# cuda
include_directories(/usr/local/cuda/include)
link_directories(/usr/local/cuda/lib64)
# cudnn
include_directories(/usr/include/x86_64-linux-gnu/)
link_directories(/usr/lib/x86_64-linux-gnu/)
# tensorrt
if ("${TRT_IN_DIRS}" STREQUAL "")
	message("use default include dir /usr/local/tensorrt/include/")
	include_directories(/usr/local/tensorrt/include/)
else()
	message("use custom include dir ${TRT_IN_DIRS}")
	include_directories(${TRT_IN_DIRS})
endif()

if ("${TRT_LIB_DIRS}" STREQUAL "")
	message("use default lib dir /usr/local/tensorrt/lib/")
	link_directories(/usr/local/tensorrt/lib/)
else()
	message("use custom lib dir ${TRT_LIB_DIRS}")
	link_directories(${TRT_LIB_DIRS})
endif()

include_directories(
    ${CUDA_INCLUDE_DIRS}
    ${PROJECT_SOURCE_DIR}/include
)

message(${CUDA_INCLUDE_DIRS})
message(${CUDA_LIBRARIES})

if ("${GPU_CC}" STREQUAL "")
	message("use default sm 89")
	set(CUDA_GEN_CODE "-gencode=arch=compute_89,code=sm_89")
else()
	message("use custom sm ${GPU_CC}")
	set(CUDA_GEN_CODE "-gencode=arch=compute_${GPU_CC},code=sm_${GPU_CC}")
endif()
#set(CUDA_GEN_CODE 
#	-gencode=arch=compute_75,code=sm_75
#	-gencode=arch=compute_80,code=sm_80
#	-gencode=arch=compute_86,code=sm_86
#	-gencode=arch=compute_89,code=sm_89)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17 -Wall -O3 -Wfatal-errors -pthread -w")
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -std=c++17 -O3 -Xcompiler -fPIC -w ${CUDA_GEN_CODE}")

#kernel
include_directories(${PROJECT_SOURCE_DIR}/kernel)
file(GLOB_RECURSE cu_files1 ${PROJECT_SOURCE_DIR}/kernel/*.cu)
set(kernelCuSo "kernelCu")
#cuda_add_library(${kernelCuSo} SHARED ${cu_files1})
cuda_add_library(${kernelCuSo} ${cu_files1})
target_link_libraries(${kernelCuSo} ${CUDA_LIBRARIES} cuda ${CUDA_cublas_LIBRARIES})

set(project1 plugin)
include_directories(${PROJECT_SOURCE_DIR}/${project1})
FILE(GLOB_RECURSE CPP_FILE1 ${PROJECT_SOURCE_DIR}/${project1}/*.cpp)
set(pluginSo "pluginSo_cusLn3d_eps6")
add_library(${pluginSo} SHARED ${CPP_FILE1})
target_link_libraries(${pluginSo} ${kernelCuSo} ${CUDA_LIBRARIES} cuda nvinfer nvinfer_plugin nvonnxparser cudnn)
