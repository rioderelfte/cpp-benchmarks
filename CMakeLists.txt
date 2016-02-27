################################################################################
#                                                                              #
# Copyright (c) 2016, Florian Sowade <f.sowade@r9e.de>                         #
#                                                                              #
# Permission to use, copy, modify, and/or distribute this software for any     #
# purpose with or without fee is hereby granted, provided that the above       #
# copyright notice and this permission notice appear in all copies.            #
#                                                                              #
# THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES     #
# WITH REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF             #
# MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR      #
# ANY SPECIAL, DIRECT, INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES       #
# WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN        #
# ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF      #
# OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.               #
#                                                                              #
################################################################################

project("benchmark" CXX)
cmake_minimum_required(VERSION 3.3)

set(CELERO_BASE_DIR "external/celero")
add_library(celero STATIC
    "${CELERO_BASE_DIR}/src/Archive.cpp"
    "${CELERO_BASE_DIR}/src/Benchmark.cpp"
    "${CELERO_BASE_DIR}/src/Callbacks.cpp"
    "${CELERO_BASE_DIR}/src/Celero.cpp"
    "${CELERO_BASE_DIR}/src/Console.cpp"
    "${CELERO_BASE_DIR}/src/Distribution.cpp"
    "${CELERO_BASE_DIR}/src/Executor.cpp"
    "${CELERO_BASE_DIR}/src/Experiment.cpp"
    "${CELERO_BASE_DIR}/src/JUnit.cpp"
    "${CELERO_BASE_DIR}/src/Print.cpp"
    "${CELERO_BASE_DIR}/src/Result.cpp"
    "${CELERO_BASE_DIR}/src/ResultTable.cpp"
    "${CELERO_BASE_DIR}/src/Statistics.cpp"
    "${CELERO_BASE_DIR}/src/TestFixture.cpp"
    "${CELERO_BASE_DIR}/src/TestVector.cpp"
    "${CELERO_BASE_DIR}/src/ThreadTestFixture.cpp"
    "${CELERO_BASE_DIR}/src/Timer.cpp"
    "${CELERO_BASE_DIR}/src/Utilities.cpp")
target_include_directories(celero PUBLIC
    "${CELERO_BASE_DIR}/include")
target_compile_features(celero PUBLIC
    "cxx_auto_type"
    "cxx_enum_forward_declarations"
    "cxx_lambdas"
    "cxx_override"
    "cxx_range_for"
    "cxx_right_angle_brackets"
    "cxx_rvalue_references"
    "cxx_strong_enums")

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/external/findTBB")
find_package(TBB REQUIRED)

add_subdirectory("sumVectorOfInts")
