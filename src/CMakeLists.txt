include_directories(.)

set_source_files_properties(${PYREDSVD_LIBRARY_SOURCE_DIR}/src/pyredsvd.pyx
    PROPERTIES CYTHON_IS_CXX TRUE )

cython_add_module(pyredsvd pyredsvd.pyx util.cpp)

