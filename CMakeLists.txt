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

# Make sure the epd headers get included, do this _after_ UseCython
set(PYTHON_INCLUDE_DIRS
    /home/mdenil/.local/epd-7.1-2-rh5-x86_64/include/python2.7
    )

#set(CMAKE_CXX_FLAGS "-Wall -O3 -I${EIGEN3_INCLUDE_DIR} -DEIGEN_DEFAULT_TO_ROW_MAJOR")
set(CMAKE_CXX_FLAGS "-Wall -O0 -ggdb -I${EIGEN3_INCLUDE_DIR} -DEIGEN_DEFAULT_TO_ROW_MAJOR")

add_subdirectory(src)



