cmake_minimum_required(VERSION 3.12)

# compiler
set(CMAKE_CXX_COMPILER          ${CXX})

# compiler flags
set(CMAKE_Fortran_COMPILER      ${FC}) 

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_BUILD_TYPE Release)

project(axisem LANGUAGES CXX Fortran)
file (GLOB SOURCE_FILES "${PROJECT_SOURCE_DIR}/src/*")
set(CMAKE_VERBOSE_MAKEFILE off)


find_package(pybind11 REQUIRED)
pybind11_add_module(libsem ${SOURCE_FILES})

install(TARGETS libsem
  COMPONENT python
  LIBRARY DESTINATION ${PROJECT_SOURCE_DIR}/lib
)