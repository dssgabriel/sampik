# Copyright (C) 2024, CEA
#
# Licensed under the MIT License ("the License" hereafter);
# You may not use this file except in compliance with the License.
# You should have received a full copy of the License along with this program;
# if not, you can obtain a copy at:
#
#    https://opensource.org/license/MIT
#
# The software is provided “as is”, without warranty of any kind, express
# or implied, including but not limited to the warranties of merchantability,
# fitness for a particular purpose and noninfringement. In no event shall
# the authors or copyright holders be liable for any claim, damages or other
# liability, whether in an action of contract, tort or otherwise, arising from,
# out of or in connection with the software or the use or other dealings
# in the software.
#
# Author: Gabriel Dos Santos <gabriel.dossantos@cea.fr>

cmake_minimum_required(VERSION 3.23)

project(sampik-examples LANGUAGES CXX DESCRIPTION "sampik examples")

if(NOT TARGET sampik::sampik)
  find_package(sampik REQUIRED)
endif()

find_package(KokkosKernels REQUIRED)

add_executable(distributed_row_matvec)
target_sources(distributed_row_matvec PRIVATE distributed_row_matvec.cpp)
target_compile_features(distributed_row_matvec PRIVATE cxx_std_20)
set_target_properties(distributed_row_matvec PROPERTIES CXX_EXTENSIONS OFF)
target_link_libraries(distributed_row_matvec PRIVATE sampik::sampik)

# TODO: implement 2D-block version of the distributed matrix-vector product
#add_executable(distributed_block_matvec)
#target_sources(distributed_block_matvec PRIVATE distributed_block_matvec.cpp)
#target_link_libraries(distributed_block_matvec PRIVATE sampik::sampik)
