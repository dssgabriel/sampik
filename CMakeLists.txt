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
# Author: Gabriel Dos Santos <gabriel.dossantos@cea.fr, dss.gabriel@protonmail.com>

cmake_minimum_required(VERSION 3.16)
project(sampik LANGUAGES CXX DESCRIPTION "SAMPIK: Simple API for MPI + Kokkos")

set(CMAKE_COLOR_DIAGNOSTICS ON)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 20)
  set(CMAKE_CXX_STANDARD_REQUIRED ON)
  message(STATUS "Setting C++ standard to C++${CMAKE_CXX_STANDARD}")
endif()

if(CMAKE_BUILD_TYPE EQUAL Debug)
    add_compile_options(-Wall -Wextra -Wpedantic -Wconversion -fanalyzer)
endif()

include(GNUInstallDirs)

find_package(MPI REQUIRED)
add_subdirectory(ext/kokkos)

enable_testing()

add_subdirectory(src)

if (CMAKE_TESTING_ENABLED)
    add_subdirectory(tests)
endif()
