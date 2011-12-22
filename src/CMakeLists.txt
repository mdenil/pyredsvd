add_subdirectory(redsvd-core)
include_directories(redsvd-core)

set_source_files_properties(${PYREDSVD_LIBRARY_SOURCE_DIR}/src/pyredsvd.pyx
    PROPERTIES CYTHON_IS_CXX TRUE )

cython_add_module(pyredsvd pyredsvd.pyx)

#add_dependencies(pyredsvd redsvd-core)
target_link_libraries(pyredsvd redsvd-core)
