cmake_minimum_required(VERSION 3.20)

project(phasebatch_worker LANGUAGES CXX)

find_package(LLVM REQUIRED CONFIG)

message(STATUS "Found LLVM ${LLVM_PACKAGE_VERSION}")
message(STATUS "Using LLVMConfig.cmake in ${LLVM_DIR}")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_executable(phasebatch-worker phasebatch_worker.cpp)
get_filename_component(PHASEBATCH_LLVM_ROOT "${LLVM_DIR}/../../../.." ABSOLUTE)
set(PHASEBATCH_LLVM_SOURCE_DIR "${PHASEBATCH_LLVM_ROOT}/llvm" CACHE PATH "LLVM source tree")
find_path(PHASEBATCH_LLVM_DIFF_INCLUDE_DIR
  DifferenceEngine.h
  HINTS "${PHASEBATCH_LLVM_SOURCE_DIR}/tools/llvm-diff/lib"
  NO_DEFAULT_PATH
  REQUIRED
)
target_include_directories(phasebatch-worker PRIVATE
  ${LLVM_INCLUDE_DIRS}
  ${PHASEBATCH_LLVM_DIFF_INCLUDE_DIR}
)
separate_arguments(PHASEBATCH_LLVM_DEFINITIONS NATIVE_COMMAND "${LLVM_DEFINITIONS}")
target_compile_options(phasebatch-worker PRIVATE ${PHASEBATCH_LLVM_DEFINITIONS})
target_compile_options(phasebatch-worker PRIVATE -Wa,-mbig-obj -fno-exceptions -fno-rtti)

llvm_map_components_to_libnames(PHASEBATCH_LLVM_LIBS
  Analysis
  BitReader
  BitWriter
  Core
  IPO
  IRReader
  InstCombine
  Passes
  ScalarOpts
  Support
  TransformUtils
  Vectorize
)

target_link_libraries(phasebatch-worker PRIVATE LLVMDiff ${PHASEBATCH_LLVM_LIBS})
target_link_options(phasebatch-worker PRIVATE -Wl,--stack,16777216)
