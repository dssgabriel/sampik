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


# --- SAMPIK TARGET --- #
add_library(sampik INTERFACE)
add_library(sampik::sampik ALIAS sampik)

# Public headers
target_sources(
  sampik
  INTERFACE
    FILE_SET sampik_public_headers
    TYPE HEADERS
    BASE_DIRS ${CMAKE_SOURCE_DIR}/include
    FILES
      ${CMAKE_SOURCE_DIR}/include/sampik/sampik.hpp
      ${CMAKE_SOURCE_DIR}/include/sampik/traits.hpp
      ${CMAKE_SOURCE_DIR}/include/sampik/concepts.hpp
      ${CMAKE_SOURCE_DIR}/include/sampik/handle.hpp
      ${CMAKE_SOURCE_DIR}/include/sampik/request.hpp
      ${CMAKE_SOURCE_DIR}/include/sampik/initialize.hpp
      ${CMAKE_SOURCE_DIR}/include/sampik/finalize.hpp
      ${CMAKE_SOURCE_DIR}/include/sampik/point_to_point.hpp
      ${CMAKE_SOURCE_DIR}/include/sampik/collective.hpp
      ${CMAKE_SOURCE_DIR}/include/sampik/utility.hpp
      ${CMAKE_SOURCE_DIR}/include/sampik/config.hpp # header generated at configuration-time
    FILE_SET sampik_detail_headers
    TYPE HEADERS
    BASE_DIRS ${CMAKE_SOURCE_DIR}/include
    FILES
      ${CMAKE_SOURCE_DIR}/include/sampik/detail/communication_spaces.hpp
      ${CMAKE_SOURCE_DIR}/include/sampik/detail/types.hpp
)

# Compilation flags
target_compile_features(sampik INTERFACE cxx_std_20)
set_target_properties(sampik PROPERTIES CXX_EXTENSIONS OFF)
target_compile_options(
  sampik
  INTERFACE
    -Wall
    -Wextra
    -Wshadow
    -Wpedantic
    -Wcast-align
    -Wconversion
    -Wmissing-include-dirs
    -Wformat=2
)

# Link dependencies
target_link_libraries(
  sampik
  INTERFACE
    MPI::MPI_CXX
    Kokkos::kokkos
)


# --- PACKAGING --- #
include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

# Config version header
set(SAMPIK_VERSION_MAJOR ${CMAKE_PROJECT_VERSION_MAJOR} CACHE STRING "" FORCE)
set(SAMPIK_VERSION_MINOR ${CMAKE_PROJECT_VERSION_MINOR} CACHE STRING "" FORCE)
set(SAMPIK_VERSION_PATCH ${CMAKE_PROJECT_VERSION_PATCH} CACHE STRING "" FORCE)
configure_file(
  ${CMAKE_SOURCE_DIR}/cmake/sampik_config.hpp.in
  ${CMAKE_SOURCE_DIR}/include/sampik/config.hpp
)

# Config version CMake file
write_basic_package_version_file(
  ${CMAKE_BINARY_DIR}/sampik-config-version.cmake
  COMPATIBILITY SameMajorVersion
  ARCH_INDEPENDENT # sampik is a header-only library
)

# Install CMake package config files
install(
  FILES
    ${CMAKE_SOURCE_DIR}/cmake/sampik-config.cmake
    ${CMAKE_BINARY_DIR}/sampik-config-version.cmake
  DESTINATION ${CMAKE_INSTALL_DATADIR}/cmake/sampik
)

# Export and install public headers
install(TARGETS sampik EXPORT sampikTargets FILE_SET sampik_public_headers FILE_SET sampik_detail_headers)
install(
  EXPORT sampikTargets
  FILE sampikTargets.cmake
  NAMESPACE sampik::
  DESTINATION ${CMAKE_INSTALL_DATADIR}/cmake/sampik
)
