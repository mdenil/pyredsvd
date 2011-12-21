cmake_minimum_required(VERSION 2.6)
project(pyresvd)


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



set(SOFTWARE_DIR "/home/mdenil/software/")
set(EIGEN_DIR "${SOFTWARE_DIR}/eigen-3.0.4")

set(CMAKE_MODULE_PATH "${EIGEN_DIR}/cmake")

############################ 
# Find libraries

set(Eigen3_FIND_VERSION 3.0.0)
find_package(Eigen3)





set(CMAKE_CXX_FLAGS "-Wall -O3")

add_subdirectory(src)
