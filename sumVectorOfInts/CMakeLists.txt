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

add_executable(
    sumVectorOfInts
    benchmark.cpp)
target_compile_features(sumVectorOfInts PRIVATE
    "cxx_alias_templates"
    "cxx_auto_type"
    "cxx_lambdas"
    "cxx_nonstatic_member_init"
    "cxx_override"
    "cxx_range_for"
    "cxx_uniform_initialization")
target_link_libraries(sumVectorOfInts PRIVATE
    ${TBB_LIBRARIES}
    celero)
target_compile_definitions(sumVectorOfInts PRIVATE
    ${TBB_DEFINITIONS})
target_include_directories(sumVectorOfInts PRIVATE
    ${TBB_INCLUDE_DIRS})
