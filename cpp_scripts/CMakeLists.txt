cmake_minimum_required(VERSION 3.10)

project(tensorrt_llm_cpp)

set(TRTLLM_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../TensorRT-LLM")
set(TRTLLM_LIB_PATH "${TRTLLM_DIR}/tensorrt_llm/libs/libtensorrt_llm.so")
set(TRTLLM_PLUGIN_PATH "${TRTLLM_DIR}/tensorrt_llm/libs/libnvinfer_plugin_tensorrt_llm.so.10")
set(TRTLLM_INCLUDE_DIR "${TRTLLM_DIR}/cpp/include")

## from examples/cpp/executor/CMakeLists.txt
execute_process(
    COMMAND bach -c "nm -f posix -$ ${TRTLLM_LIB_PATH} | grep __cxx11"
    RESULT_VARIABLE GLIB_CXX11_FOUND
    OUTPUT_QUIET)
if(GLIB_CXX11_FOUND EQUAL 0)
    set(USE_CXX11_ABI 1)
else()
    set(USE_CXX11_ABI 0)
endif()
add_compile_definitions("-D_GLIBCXX_USE_CXX11_ABI=${USE_CXX11_ABI}")
##

# tensorrt
if(NOT DEFINED TRT_DIR)
    message(FATAL_ERROR "The root of tensorrt -DTRT_DIR is not defined.")
endif()
set(TRT_INCLUDE_DIR "${TRT_DIR}/include")
set(TRT_LIB_PATH "${TRT_DIR}/lib/libnvinfer.so")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)

find_package(CUDAToolkit REQUIRED)

# tensorrt_llm shared lib
add_library(tensorrt_llm SHARED IMPORTED)
set_property(TARGET tensorrt_llm PROPERTY IMPORTED_LOCATION ${TRTLLM_LIB_PATH})
set_property(
    TARGET tensorrt_llm PROPERTY IMPORTED_LINK_INTERFACE_LIBRARIES
    CUDA::cuda_driver CUDA::cudart_static CUDA::nvml
)

# nvinfer_plugin_tensorrt_llm shared lib
add_library(nvinfer_plugin_tensorrt_llm SHARED IMPORTED)
set_property(
    TARGET nvinfer_plugin_tensorrt_llm PROPERTY IMPORTED_LOCATION ${TRTLLM_PLUGIN_PATH}
)
set_property(
    TARGET nvinfer_plugin_tensorrt_llm PROPERTY IMPORTED_LINK_INTERFACE_LIBRARIES 
    tensorrt_llm
)

include_directories(${TRT_INCLUDE_DIR} ${TRTLLM_INCLUDE_DIR} ${CUDA_INCLUDE_DIRS})

find_package(MPI REQUIRED)

add_executable(main main.cpp)
target_link_libraries(main nvinfer_plugin_tensorrt_llm)
