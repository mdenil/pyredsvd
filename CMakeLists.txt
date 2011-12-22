cmake_minimum_required(VERSION 2.8)
project(PYREDSVD_LIBRARY)


# ---------- Setup output Directories -------------------------
SET (CMAKE_LIBRARY_OUTPUT_DIRECTORY
   ${PROJECT_BINARY_DIR}/lib
   CACHE PATH
   "Single Directory for all Libraries"
   )

# --------- Setup the Executable output Directory -------------
SET (CMAKE_RUNTIME_OUTPUT_DIRECTORY
   ${PROJECT_BINARY_DIR}/bin
   CACHE PATH
   "Single Directory for all Executables."
   )

# --------- Setup the Executable output Directory -------------
SET (CMAKE_ARCHIVE_OUTPUT_DIRECTORY
   ${PROJECT_BINARY_DIR}/lib
   CACHE PATH
   "Single Directory for all static libraries."
   )



set(CMAKE_MODULE_PATH
    ${CMAKE_MODULE_PATH}
    ${PYREDSVD_LIBRARY_SOURCE_DIR}/cmake
    )

############################ 
# Find libraries

set(Eigen3_FIND_VERSION 3.0.0)
find_package(Eigen3)

include(UseCython)

# If we have EPD make sure we use its headers instead of
# the system python headers.  Do this after UseCython.
if(NOT "$ENV{EPD_ROOT}" STREQUAL "")
    set(PYTHON_INCLUDE_DIRS $ENV{EPD_ROOT}/include/python2.7)
endif()

#set(CMAKE_CXX_FLAGS "-Wall -O3 -I${EIGEN3_INCLUDE_DIR} -DEIGEN_DEFAULT_TO_ROW_MAJOR")
set(CMAKE_CXX_FLAGS "-Wall -O0 -ggdb -I${EIGEN3_INCLUDE_DIR} -DEIGEN_DEFAULT_TO_ROW_MAJOR")

add_subdirectory(src)

