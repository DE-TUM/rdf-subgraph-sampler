cmake_minimum_required(VERSION 3.20)
project(bgpgen LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

option(BGPGEN_NATIVE "Optimize for the host CPU (-march=native)" ON)
option(BGPGEN_TESTS "Build tests" ON)
option(BGPGEN_PYTHON "Build the Python module (used by pip, see ../pyproject.toml)" OFF)

if(BGPGEN_PYTHON)
  # The static core is linked into a shared module.
  set(CMAKE_POSITION_INDEPENDENT_CODE ON)
endif()

find_package(ZLIB)
# Needed at link time on systems with glibc older than 2.34.
set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)

function(bgpgen_set_warnings target)
  if(MSVC)
    target_compile_options(${target} PRIVATE /W4)
  else()
    target_compile_options(${target} PRIVATE -Wall -Wextra -Wpedantic)
    if(BGPGEN_NATIVE)
      target_compile_options(${target} PRIVATE
        $<$<CONFIG:Release>:-march=native>)
    endif()
  endif()
endfunction()

# --- Core library -----------------------------------------------------------
add_library(bgpgen_core STATIC
  src/term_table.cpp
  src/graph.cpp
  src/nt_parser.cpp
  src/gz_reader.cpp
  src/subject_index.cpp
  src/cardinality.cpp
  src/dedup.cpp
  src/sampler_star.cpp
  src/template.cpp
  src/matcher.cpp
  src/match_weights.cpp
  src/sampler_complex.cpp
  src/workload.cpp
)
target_include_directories(bgpgen_core PUBLIC include)
target_link_libraries(bgpgen_core PUBLIC Threads::Threads)
if(ZLIB_FOUND)
  target_compile_definitions(bgpgen_core PRIVATE BGPGEN_HAVE_ZLIB)
  target_link_libraries(bgpgen_core PRIVATE ZLIB::ZLIB)
endif()
bgpgen_set_warnings(bgpgen_core)

# --- CLI ---------------------------------------------------------------------
add_executable(bgpgen cli/main.cpp)
target_link_libraries(bgpgen PRIVATE bgpgen_core)
bgpgen_set_warnings(bgpgen)

# --- Python module -------------------------------------------------------------
if(BGPGEN_PYTHON)
  find_package(Python 3.9 COMPONENTS Interpreter Development.Module REQUIRED)
  find_package(nanobind CONFIG REQUIRED)
  nanobind_add_module(_bgpgen NB_STATIC bindings/python/module.cpp)
  target_link_libraries(_bgpgen PRIVATE bgpgen_core)
  install(TARGETS _bgpgen LIBRARY DESTINATION bgpgen)
  if(SKBUILD)
    # pip also installs the command line tool.
    install(TARGETS bgpgen RUNTIME DESTINATION ${SKBUILD_SCRIPTS_DIR})
  endif()
endif()

# --- Evaluation driver for the sampling options (bench/) ---------------------
add_executable(bgpgen_eval bench/eval_improvements.cpp)
target_link_libraries(bgpgen_eval PRIVATE bgpgen_core)
bgpgen_set_warnings(bgpgen_eval)

# --- Counts for queries from a file (labels and cross-checks other workloads) --
add_executable(bgpgen_count bench/count_queries.cpp)
target_link_libraries(bgpgen_count PRIVATE bgpgen_core)
bgpgen_set_warnings(bgpgen_count)

# --- Tests ---------------------------------------------------------------------
if(BGPGEN_TESTS)
  enable_testing()
  add_executable(bgpgen_tests
    tests/main.cpp
    tests/test_load.cpp
    tests/test_star.cpp
    tests/test_template.cpp
    tests/test_dedup.cpp
    tests/test_cardinality.cpp
    tests/test_weights.cpp
  )
  target_link_libraries(bgpgen_tests PRIVATE bgpgen_core)
  target_include_directories(bgpgen_tests PRIVATE third_party/doctest)
  bgpgen_set_warnings(bgpgen_tests)
  add_test(NAME unit COMMAND bgpgen_tests)
endif()
