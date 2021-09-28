# structure and content of CMakeLists.txt files adapted from 
# https://stackoverflow.com/questions/55635294/how-to-create-packages-with-cmake

cmake_minimum_required(VERSION 3.9.0)
project(parallelproj LANGUAGES CUDA CXX VERSION 0.8.0)

include(CMakePackageConfigHelpers)
include(CheckLanguage)

# get standard paths for installation
include(GNUInstallDirs)

set (CMAKE_BUILD_TYPE Release CACHE STRING "build type" FORCE)

#-------------------------------------------------------------

set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS TRUE)

#-------------------------------------------------------------
# find the required OpeMP
find_package(OpenMP REQUIRED)


#-------------------------------------------------------------

# auto detect the cuda architecture
# https://stackoverflow.com/questions/68223398/how-can-i-get-cmake-to-automatically-detect-the-value-for-cuda-architectures
include(FindCUDA/select_compute_arch)
CUDA_DETECT_INSTALLED_GPUS(INSTALLED_GPU_CCS_1)
string(STRIP "${INSTALLED_GPU_CCS_1}" INSTALLED_GPU_CCS_2)
string(REPLACE " " ";" INSTALLED_GPU_CCS_3 "${INSTALLED_GPU_CCS_2}")
string(REPLACE "." "" CUDA_ARCH_LIST "${INSTALLED_GPU_CCS_3}")
set(CMAKE_CUDA_ARCHITECTURES "${CUDA_ARCH_LIST}")

# add openmp flags to cuda compiler
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler=${OpenMP_CXX_FLAGS}") 
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_CXX_FLAGS}") 


#--------------------------------------------------------------------------------------------------

file(GLOB CUDA_SOURCES src/*.cu)

add_library(parallelproj_cuda SHARED ${CUDA_SOURCES})
add_library(parallelproj::parallelproj_cuda ALIAS parallelproj_cuda)
set_target_properties(parallelproj_cuda PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(parallelproj_cuda PROPERTIES PUBLIC_HEADER "include/parallelproj_cuda.h"
                                                   SOVERSION ${parallelproj_VERSION})

target_include_directories(parallelproj_cuda PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/>
  $<INSTALL_INTERFACE:include/>
  )

# we have to add the openmp library to be linked
target_link_libraries(parallelproj_cuda PUBLIC OpenMP::OpenMP_CXX)

install(TARGETS parallelproj_cuda EXPORT parallelprojTargets 
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR} 
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR} 
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
        PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
