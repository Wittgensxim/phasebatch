cmake_minimum_required(VERSION 3.20)

project(advisor_pair_scale_2n_merge_helper LANGUAGES CXX)

find_package(LLVM REQUIRED CONFIG)

message(STATUS "Found LLVM ${LLVM_PACKAGE_VERSION}")
message(STATUS "Using LLVMConfig.cmake in ${LLVM_DIR}")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_executable(phasebatch-2n-merge phasebatch_2n_merge.cpp)
target_include_directories(phasebatch-2n-merge PRIVATE ${LLVM_INCLUDE_DIRS})

separate_arguments(ADVISOR_2N_LLVM_DEFINITIONS NATIVE_COMMAND "${LLVM_DEFINITIONS}")
target_compile_options(phasebatch-2n-merge PRIVATE
  ${ADVISOR_2N_LLVM_DEFINITIONS}
  -Wa,-mbig-obj
  -fno-exceptions
  -fno-rtti
)

llvm_map_components_to_libnames(ADVISOR_2N_LLVM_LIBS
  Core
  IRReader
  Support
  TransformUtils
)
target_link_libraries(phasebatch-2n-merge PRIVATE ${ADVISOR_2N_LLVM_LIBS})
target_link_options(phasebatch-2n-merge PRIVATE -Wl,--stack,16777216)
