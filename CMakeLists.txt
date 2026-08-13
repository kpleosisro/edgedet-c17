cmake_minimum_required(VERSION 3.20)
project(edgedet VERSION 0.1.0 LANGUAGES C)

set(CMAKE_C_STANDARD 17)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS OFF)

option(ED_BUILD_TESTS "Build native tests" ON)
option(ED_ENABLE_NATIVE "Enable host-specific SIMD flags" OFF)
option(ED_ENABLE_SIMD "Build architecture-specific runtime-dispatched kernels" ON)

add_library(edgedet STATIC
    src/ed_crc32.c
    src/ed_sha256.c
    src/ed_io.c
    src/ed_mapped_file.c
    src/ed_parallel.c
    src/ed_power.c
    src/ed_image.c
    src/ed_ops.c
    src/ed_graph.c
    src/ed_picodet_graph.c
    src/ed_model.c
    src/ed_calibrate.c
    src/ed_train.c
    src/ed_eval.c
)
if(ED_ENABLE_SIMD AND CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64|amd64|i.86")
  target_sources(edgedet PRIVATE src/ed_ops_avx2.c)
  target_compile_definitions(edgedet PRIVATE ED_HAVE_AVX2_IMPL=1)
  if(MSVC)
    set_source_files_properties(src/ed_ops_avx2.c PROPERTIES COMPILE_OPTIONS "/arch:AVX2")
  else()
    set_source_files_properties(src/ed_ops_avx2.c PROPERTIES COMPILE_OPTIONS "-mavx2;-mfma")
  endif()
endif()
if(ED_ENABLE_SIMD AND CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|AARCH64|arm64|ARM64|armv7|ARMv7")
  target_sources(edgedet PRIVATE src/ed_ops_neon.c)
  target_compile_definitions(edgedet PRIVATE ED_HAVE_NEON_IMPL=1)
  if(NOT MSVC AND CMAKE_SYSTEM_PROCESSOR MATCHES "armv7|ARMv7")
    set_source_files_properties(src/ed_ops_neon.c PROPERTIES COMPILE_OPTIONS "-mfpu=neon")
  endif()
endif()
target_include_directories(edgedet PUBLIC include PRIVATE src third_party)
target_compile_definitions(edgedet PRIVATE _CRT_SECURE_NO_WARNINGS)
find_package(Threads REQUIRED)
target_link_libraries(edgedet PUBLIC Threads::Threads)
if(WIN32)
  target_link_libraries(edgedet PUBLIC psapi)
endif()

if(MSVC)
  target_compile_options(edgedet PRIVATE /W4)
  set_source_files_properties(src/ed_image.c PROPERTIES COMPILE_OPTIONS "/w")
else()
  target_compile_options(edgedet PRIVATE -Wall -Wextra -Wpedantic -Wconversion -Wshadow)
  set_source_files_properties(src/ed_image.c PROPERTIES COMPILE_OPTIONS "-w")
  target_link_libraries(edgedet PUBLIC m)
  if(ED_ENABLE_NATIVE)
    target_compile_options(edgedet PRIVATE -march=native)
  endif()
endif()

foreach(tool edtrain edeval edinfer edpack-voc edpack-yolo edpack-manifest edremap edcompact edinspect edbench edsplit edviz)
  string(REPLACE "-" "_" target ${tool})
  add_executable(${target} tools/${target}.c)
  target_link_libraries(${target} PRIVATE edgedet)
  target_include_directories(${target} PRIVATE src)
  set_target_properties(${target} PROPERTIES OUTPUT_NAME ${tool})
endforeach()

if(ED_BUILD_TESTS)
  enable_testing()
  add_executable(test_edgedet tests/test_edgedet.c)
  target_link_libraries(test_edgedet PRIVATE edgedet)
  target_include_directories(test_edgedet PRIVATE src)
  target_compile_definitions(test_edgedet PRIVATE
    ED_TEST_MODEL_PATH="${CMAKE_SOURCE_DIR}/models/picodet_s_coco80_fp16.edm")
  add_test(NAME edgedet_core COMMAND test_edgedet)
  add_executable(dump_picodet tests/dump_picodet.c)
  target_link_libraries(dump_picodet PRIVATE edgedet)
  target_include_directories(dump_picodet PRIVATE src)
endif()
