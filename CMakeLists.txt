cmake_minimum_required(VERSION 3.24)
project(tmrcacu LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

# Default to sm_80 (A100) + sm_100 (B200 Blackwell)
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 80)
endif()

# Performance flags: --use_fast_math gives ~10-15% on transcendentals
# (acceptable for Gamma-SMC where fast_pow10 is already approximate)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --use_fast_math")

# Find pybind11
find_package(pybind11 REQUIRED)
find_package(Python REQUIRED COMPONENTS Interpreter Development)

# Include directories
include_directories(${CMAKE_SOURCE_DIR}/include)

# Build as a shared library instead of static to avoid device link issues
add_library(tmrcacu_kernels SHARED
    src/kernels/bitpack.cu
    src/kernels/sfs.cu
    src/kernels/hmm_forward_backward.cu
    src/kernels/tier1_divergence.cu
    src/kernels/tier2_pelt.cu
    src/kernels/ultrametric.cu
    src/kernels/convergence.cu
    src/kernels/adaptive_prior.cu
    src/kernels/gamma_smc.cu
    src/kernels/gamma_smc_flow.cu
    src/demography.cpp
    src/flow_field.cpp
)
target_include_directories(tmrcacu_kernels PUBLIC ${CMAKE_SOURCE_DIR}/include)
set_target_properties(tmrcacu_kernels PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    POSITION_INDEPENDENT_CODE ON
)
target_link_libraries(tmrcacu_kernels PRIVATE cudart)

# Python module
pybind11_add_module(_core src/bindings.cpp)
target_link_libraries(_core PRIVATE tmrcacu_kernels)
target_include_directories(_core PRIVATE ${CMAKE_SOURCE_DIR}/include ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

# Install the Python module into python/tmrca_cu/
install(TARGETS _core DESTINATION ${CMAKE_SOURCE_DIR}/python/tmrca_cu)
